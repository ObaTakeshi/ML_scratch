module Functional
    using Statistics

    function sigmoid(x)
        return 1.0 ./ (1.0 .+ exp.(-x))
    end

    function softmax(a::AbstractVector{T}) where T
        c = maximum(a)
        exp_a = exp.(a .- c)
        return exp_a ./ sum(exp_a)
    end

    function softmax(a::AbstractMatrix{T}) where T
        mapslices(softmax, a, dims=1)
    end

    function tanh(x::AbstractMatrix{T}) where T
        x1 = exp.(x)
        x2 = exp.(-x)
        (x1 .- x2) ./ (x1 .+ x2)
    end

    function onehot(::Type{T}, t::AbstractVector, l::AbstractVector) where T
        r = zeros(T, length(l), length(t))
        for i = 1:length(t)
            r[findfirst(isequal(t[i]), l), i] = 1
        end
        return r
    end
    @inline onehot(t, l) = onehot(Int, t, l)

    function BCELoss(y, t)
        delta = 1e-7
        return mean(-t .* log.(y .+ delta) .- (1 .- t) .* log.(1 .- y .+ delta))
    end

    function crossEntropyLoss(y::Vector, t::Vector)
        delta = 1e-7
        return -(t' * log.(y .+ delta)) / size(y, 2)
    end

    function crossEntropyLoss(y::Matrix, t::Matrix)
        batch_size = size(y, 2)
        delta = 1e-7
        return -sum(t .* log.(y .+ delta)) / batch_size
    end
end
