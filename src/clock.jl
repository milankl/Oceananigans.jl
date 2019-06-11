mutable struct Clock{T<:Number}
  time::T
  timevec::Array{T,1}
  iteration::Int
  Nt::Int
  Δt::T
  t_start::Real   # time when the simulation starts
end

function Clock(T::DataType,start_time::Number,iteration::Int,Nt::Int,Δt::Number)
    timevec = Array{T}(start_time:Δt:Δt*Nt)
    t_start = 0.0
    Clock{T}(start_time, timevec, iteration, Nt, Δt, t_start)
end


## Feedback functions
"""Converts time step into percent for feedback."""
function progress(i::Int,Nt::Int)
    if ((i+1)/Nt*100 % 1) < (i/Nt*100 % 1)  # update every 1 percent steps.
        percent = Int(round((i+1)/Nt*100))
        print("\r\u1b[K")   # remove previous percentage
        print("$percent%")  # print new one
    end
end

"""Returns a human readable string representing seconds in terms of days, hours, minutes or seconds."""
function readable_secs(secs::Real)
    days = Int(floor(secs/3600/24))
    hours = Int(floor((secs/3600) % 24))
    minutes = Int(floor((secs/60) % 60))
    seconds = Int(floor(secs%3600%60))

    if days > 0
        return "$(days)d, $(hours)h"
    elseif hours > 0
        return "$(hours)h, $(minutes)min"
    elseif minutes > 0
        return "$(minutes)min, $(seconds)s"
    else
        return "$(seconds)s"
    end
end

"""Estimates the total time the model integration will take."""
function duration_estimate(i::Int,Nt::Int,t_start::Real)
    time_per_step = (time()-t_start)/i
    time_total = Int(round(time_per_step*Nt))
    time_to_go = Int(round(time_per_step*(Nt-i)))

    s1 = "Model integration will take approximately "*readable_secs(time_total)*","
    s2 = "and is hopefully done on "*Dates.format(now() + Dates.Second(time_to_go),Dates.RFC1123Format)

    println(s1)     # print inline
    println(s2)
end

"""Feedback function that calls duration estimate, nan_detection and progress."""
function feedback(clock)

    i = clock.iteration
    Nt = clock.Nt

    if i == 1 # measure time after time loop executed once
        clock.t_start = time()
    elseif i == 100  # estimate time after 10 iterations
        duration_estimate(i,Nt,clock.t_start)
    end

    if i > 100      # show percentage only after duration is estimated
        progress(i,Nt)
    end

    if i == Nt
        s = " Integration done in "*readable_secs(time()-clock.t_start)*"."
        println(s)
    end

    return nothing
end
