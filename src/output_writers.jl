import JLD
using Distributed

using NetCDF

"A type for writing checkpoints."
struct Checkpointer <: OutputWriter
    dir::AbstractString
    filename_prefix::AbstractString
    output_frequency::Int
    padding::Int
end

"A type for writing NetCDF output."
mutable struct NetCDFOutputWriter <: OutputWriter
    dir::AbstractString
    filename_prefix::AbstractString
    output_frequency::Int
    padding::Int
    naming_scheme::Symbol
    compression::Int
    async::Bool
    onefile::Bool
    nctype::Int     # NC_FLOAT (32bit), NC_DOUBLE (64bit) are denoted with integers ?!
    file_id::Int
end

"A type for writing Binary output."
mutable struct BinaryOutputWriter <: OutputWriter
    dir::AbstractString
    filename_prefix::AbstractString
    output_frequency::Int
    padding::Int
end

function Checkpointer(; dir=".", prefix="", frequency=1, padding=9)
    Checkpointer(dir, prefix, frequency, padding)
end

function NetCDFOutputWriter(; dir=".", prefix="", frequency=1, padding=4,
                              naming_scheme=:file_number, compression=3, async=false,
                              onefile=false, nctype=NC_FLOAT, file_id=0)
    NetCDFOutputWriter(dir, prefix, frequency, padding, naming_scheme, compression, async, onefile, nctype, file_id)
end

"Return the filename extension for the `OutputWriter` filetype."
ext(fw::OutputWriter) = throw("Not implemented.")
ext(fw::NetCDFOutputWriter) = ".nc"
ext(fw::Checkpointer) = ".jld"

filename(fw::Checkpointer, iteration) = fw.filename_prefix * "model_checkpoint_" * lpad(iteration, fw.padding, "0") * ext(fw)

function filename(fw, name, iteration)
    if fw.onefile
        if iteration == 0
            runlist = filter(x->startswith(x,fw.filename_prefix),readdir(fw.dir))
            if length(runlist) > 0 # else use default value 0 for fw.file_id
                fw.file_id = maximum([parse(Int,id[end-6:end-3]) for id in runlist])+1
            end
        end
        fw.filename_prefix * name * lpad(fw.file_id,fw.padding,"0") * ext(fw)
    else
        if fw.naming_scheme == :iteration
            fw.filename_prefix * name * lpad(iteration, fw.padding, "0") * ext(fw)
        elseif fw.naming_scheme == :file_number
            file_num = Int(iteration / fw.output_frequency)
            fw.filename_prefix * name * lpad(file_num, fw.padding, "0") * ext(fw)
        else
            throw(ArgumentError("Invalid naming scheme: $(fw.naming_scheme)"))
        end
    end
end

#
# Checkpointing functions
#

function write_output(model::Model{arch}, chk::Checkpointer) where arch <: Architecture
    filepath = joinpath(chk.dir, filename(chk, model.clock.iteration))

    forcing_functions = model.forcing

    # Do not include forcing functions and FFT plans. We want to avoid serializing
    # FFTW and CuFFT plans as serializing functions is not supported by JLD, and
    # seems like a tricky business in general.
    model.forcing = nothing
    model.poisson_solver = nothing

    println("WARNING: Forcing functions will not be serialized!")

    println("[Checkpointer] Serializing model to disk: $filepath")
    f = JLD.jldopen(filepath, "w", compress=true)
    JLD.@write f model
    close(f)

    println("[Checkpointer] Reconstructing FFT plans...")
    model.poisson_solver = init_poisson_solver(arch(), model.grid, model.stepper_tmp.fCC1)

    # Putting back in the forcing functions.
    model.forcing = forcing_functions

    return nothing
end

function restore_from_checkpoint(filepath)
    println("Deserializing model from disk: $filepath")
    f = JLD.jldopen(filepath, "r")
    model = read(f, "model");
    close(f)

    println("Reconstructing FFT plans...")
    model.poisson_solver = init_poisson_solver(arch(model)(), model.grid, model.stepper_tmp.fCC1)

    model.forcing = Forcing(nothing, nothing, nothing, nothing, nothing)
    println("WARNING: Forcing functions have been set to nothing!")

    return model
end


#
# Binary output function
#

function write_output(model::Model, fw::BinaryOutputWriter)
    for (field, field_name) in zip(fw.fields, fw.field_names)
        filepath = joinpath(fw.dir, filename(fw, field_name, model.clock.iteration))

        println("[BinaryOutputWriter] Writing $field_name to disk: $filepath")
        if model.metadata == :CPU
            write(filepath, field.data)
        elseif model.metadata == :GPU
            write(filepath, Array(field.data))
        end
    end
end

function read_output(model::Model, fw::BinaryOutputWriter, field_name, time)
    filepath = joinpath(fw.dir, filename(fw, field_name, time_step))
    arr = zeros(model.metadata.float_type, model.grid.Nx, model.grid.Ny, model.grid.Nz)

    open(filepath, "r") do fio
        read!(fio, arr)
    end

    return arr
end

#
# NetCDF output functions
#
# Eventually, we want to permit the user to flexibly define what is outputted.
# The user-defined function that produces output may involve computations, launching kernels,
# etc; so this API needs to be designed. For now, we simply save u, v, w, and T.

function write_output(model::Model, fw::NetCDFOutputWriter)
    fields = Dict(
        "xC" => collect(model.grid.xC),
        "yC" => collect(model.grid.yC),
        "zC" => collect(model.grid.zC),
        "xF" => collect(model.grid.xF),
        "yF" => collect(model.grid.yF),
        "zF" => collect(model.grid.zF),
        "t" => collect(model.clock.timevec[1:fw.output_frequency:end]),
        "u" => Array(model.velocities.u.data),
        "v" => Array(model.velocities.v.data),
        "w" => Array(model.velocities.w.data),
        "T" => Array(model.tracers.T.data),
        "S" => Array(model.tracers.S.data)
    )

    if fw.async
        # Execute asynchronously on worker 2.
        @async remotecall(write_output_netcdf, 2, fw, fields, model.clock.iteration)
    else
        write_output_netcdf(fw, fields, model.clock.iteration)
    end
    return nothing
end

function write_output_netcdf(fw::NetCDFOutputWriter, fields, iteration)

    xC, yC, zC = fields["xC"], fields["yC"], fields["zC"]
    xF, yF, zF = fields["xF"], fields["yF"], fields["zF"]
    t = fields["t"]

    u, v, w = fields["u"], fields["v"], fields["w"]
    T, S    = fields["T"], fields["S"]

    filepath = joinpath(fw.dir, filename(fw, "", iteration))

    if iteration == 0 || ~fw.onefile    # initialise for onefile or output each timestep as single file

        xC_attr = Dict("longname" => "Locations of the cell centers in the x-direction.", "units" => "m")
        yC_attr = Dict("longname" => "Locations of the cell centers in the y-direction.", "units" => "m")
        zC_attr = Dict("longname" => "Locations of the cell centers in the z-direction.", "units" => "m")

        xF_attr = Dict("longname" => "Locations of the cell faces in the x-direction.", "units" => "m")
        yF_attr = Dict("longname" => "Locations of the cell faces in the y-direction.", "units" => "m")
        zF_attr = Dict("longname" => "Locations of the cell faces in the z-direction.", "units" => "m")

        t_attr = Dict("longname" => "time", "units" => "s")

        u_attr = Dict("longname" => "Velocity in the x-direction", "units" => "m/s")
        v_attr = Dict("longname" => "Velocity in the y-direction", "units" => "m/s")
        w_attr = Dict("longname" => "Velocity in the z-direction", "units" => "m/s")
        T_attr = Dict("longname" => "Temperature", "units" => "K")
        S_attr = Dict("longname" => "Salinity", "units" => "g/kg")

        isfile(filepath) && rm(filepath)

        if fw.async
            println("[Worker $(Distributed.myid()): NetCDFOutputWriter] Writing fields to disk: $filepath")
        else
            println("[NetCDFOutputWriter] Writing fields to disk: $filepath")
        end

        # nccreate(filepath, "u", "xF", xC, xC_attr,
        #                         "yC", yC, yC_attr,
        #                         "zC", zC, zC_attr,
        #                         "t", t, t_attr,
        #                         atts=u_attr, compress=fw.compression,
        #                         t=fw.nctype)
        #
        # nccreate(filepath, "v", "xC", xC, xC_attr,
        #                         "yF", yC, yC_attr,
        #                         "zC", zC, zC_attr,
        #                         "t", t, t_attr,
        #                         atts=v_attr, compress=fw.compression,
        #                         t=fw.nctype)
        #
        # nccreate(filepath, "w", "xC", xC, xC_attr,
        #                         "yC", yC, yC_attr,
        #                         "zF", zC, zC_attr,
        #                         "t", t, t_attr,
        #                         atts=w_attr, compress=fw.compression,
        #                         t=fw.nctype)

        nccreate(filepath, "T", "xC", xC, xC_attr,
                                "yC", yC, yC_attr,
                                "zC", zC, zC_attr,
                                "t", t, t_attr,
                                atts=T_attr, compress=fw.compression,
                                t=fw.nctype)

        # nccreate(filepath, "S", "xC", xC, xC_attr,
        #                         "yC", yC, yC_attr,
        #                         "zC", zC, zC_attr,
        #                         "t", t, t_attr,
        #                         atts=S_attr, compress=fw.compression,
        #                         t=fw.nctype)

        # ncwrite(u, filepath, "u", start=[1,1,1,1], count=[-1,-1,-1,1])
        # ncwrite(v, filepath, "v", start=[1,1,1,1], count=[-1,-1,-1,1])
        # ncwrite(w, filepath, "w", start=[1,1,1,1], count=[-1,-1,-1,1])
        ncwrite(T, filepath, "T", start=[1,1,1,1], count=[-1,-1,-1,1])
        # ncwrite(S, filepath, "S", start=[1,1,1,1], count=[-1,-1,-1,1])

        if fw.onefile
            #TODO
            # store ncfiles in fw and keep open
        else
            ncclose(filepath)
        end

    else # onefile output: append to existing file
        i = iteration
        freq = fw.output_frequency
        # ncwrite(u, filepath, "u", start=[1,1,1,Int(i/freq)+1], count=[-1,-1,-1,1])
        # ncwrite(v, filepath, "v", start=[1,1,1,Int(i/freq)+1], count=[-1,-1,-1,1])
        # ncwrite(w, filepath, "w", start=[1,1,1,Int(i/freq)+1], count=[-1,-1,-1,1])
        ncwrite(T, filepath, "T", start=[1,1,1,Int(i/freq)+1], count=[-1,-1,-1,1])
        # ncwrite(S, filepath, "S", start=[1,1,1,Int(i/freq)+1], count=[-1,-1,-1,1])
    end

    return nothing
end

function read_output(fw::NetCDFOutputWriter, field_name, iter)
    filepath = joinpath(fw.dir, filename(fw, "", iter))
    println("[NetCDFOutputWriter] Reading fields from disk: $filepath")

    if fw.onefile
        #TODO read only timestep iter
        field_data = ncread(filepath, field_name)
    else
        field_data = ncread(filepath, field_name)
    end

    ncclose(filepath)
    return field_data
end
