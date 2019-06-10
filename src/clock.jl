mutable struct Clock{T<:Number}
  time::T
  timevec::Array{T,1}
  iteration::Int
  Nt::Int
  Δt::T
end

function Clock(T::DataType,start_time::Number,iteration::Int,Nt::Int,Δt::Number)
    timevec = Array{T}(start_time:Δt:Δt*Nt)
    Clock{T}(start_time, timevec, iteration, Nt, Δt)
end
