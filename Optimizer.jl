module Optimizer
    include("./Layer.jl")

    mutable struct SGD
        lr::Float64
        net
        function (::Type{SGD})(net, learning_rate::Float64)
            opt = new()
            opt.net = net
            opt.lr = learning_rate
            opt
        end
    end

    function step(opt::SGD, grads)
        opt.net.params .-= opt.lr .* grads
        Main.setparams(opt.net, opt.net.params)
    end

    mutable struct Adam
        lr
        beta1
        beta2
        iter
        m
        v
        net
        function(::Type{Adam})(net, lr::Float64, beta1::Float64=0.2, beta2::Float64=0.999)
            opt = new()
            opt.net = net
            opt.lr = lr
            opt.beta1 = beta1
            opt.beta2 = beta2
            opt.iter = Int(0)
            opt.m = nothing
            opt.v = nothing
            opt
        end
    end

    function step(opt::Adam, grads)
        if opt.m == nothing
            opt.m, opt.v = Dict(), Dict()
            for (key, val) in zip(keys(opt.net.params), values(opt.net.params))
                opt.m[key] = zeros(size(val))
                opt.v[key] = zeros(size(val))
            end
        end

        opt.iter += 1
        lr_t = opt.lr * sqrt(1.0 - opt.beta2 ^ opt.iter) / (1.0 - opt.beta1 ^ opt.iter)
        for key in keys(opt.net.params)
            opt.m[key] += (1 - opt.beta1) .* (grads[key] .- opt.m[key])
            opt.v[key] += (1 - opt.beta2) .* (grads[key].^2 .- opt.v[key])
            opt.net.params[key] -= lr_t * opt.m[key] ./ (sqrt.(opt.v[key]) .+ 1e-7)
        end
        Main.setparams(opt.net, opt.net.params)
    end
end
