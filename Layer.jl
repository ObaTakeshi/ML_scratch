module Layer
    using Random
    using Statistics
    include("./Functional.jl")

    abstract type AbstractLayer end

# Affine Layer

    mutable struct AffineLayer{T} <: AbstractLayer
        W::AbstractMatrix{T}
        b::AbstractVector{T}
        x::AbstractArray{T}
        dW::AbstractMatrix{T}
        db::AbstractVector{T}
        function (::Type{AffineLayer})(W::AbstractMatrix{T}, b::AbstractVector{T}) where T
            lyr = new{T}()
            lyr.W = W
            lyr.b = b
            lyr
        end
    end

    function forward(lyr::AffineLayer{T}, x::AbstractArray{T}) where T
        lyr.x = x
        lyr.W * x .+ lyr.b
    end

    function backward(lyr::AffineLayer{T}, dout::AbstractArray{T}) where T
        dx = lyr.W' * dout
        lyr.dW = dout * lyr.x'
        lyr.db = _sumvec(dout)
        dx
    end

    @inline _sumvec(dout::AbstractVector{T}) where {T} = dout
    @inline _sumvec(dout::AbstractMatrix{T}) where {T} = vec(mapslices(sum, dout, dims=2))
    @inline _sumvec(dout::AbstractArray{T,N}) where {T,N} = vec(mapslices(sum, dout, dims=2:N))

# Activation Layer

    mutable struct SigmoidLayer <: AbstractLayer
        t
        SigmoidLayer() = new()
    end

    function forward(lyr::SigmoidLayer, x::AbstractArray{T}) where T
        t = lyr.t = Functional.sigmoid(x)
        return t
    end

    function backward(lyr::SigmoidLayer, dout::AbstractArray{T}) where T
        return dout .* lyr.t .* (1 .- lyr.t)
    end

    mutable struct ReluLayer <: AbstractLayer
        mask::AbstractArray{Bool}
        ReluLayer() = new()
    end

    function forward(lyr::ReluLayer, x::AbstractArray{T}) where T
        mask = lyr.mask = (x .<= 0)
        out = copy(x)
        out[mask] .= zero(T)
        out
    end

    function backward(lyr::ReluLayer, dout::AbstractArray{T}) where T
        dout[lyr.mask] .= zero(T)
        dout
    end

    mutable struct LeakyReluLayer <: AbstractLayer
        mask::AbstractArray{Bool}
        LeakyReluLayer() = new()
    end

    function forward(lyr::LeakyReluLayer, x::AbstractArray{T}) where T
        mask = lyr.mask = (x .< 0)
        out = copy(x)
        out[mask] .*= 0.2 
        out
    end

    function backward(lyr::LeakyReluLayer, dout::AbstractArray{T}) where T
        dout[lyr.mask] .*= 0.2
        dout
    end

    mutable struct TanhLayer <: AbstractLayer
        t
        TanhLayer() = new()
    end

    function forward(lyr::TanhLayer, x::AbstractArray{T}) where T
        t = lyr.t = Functional.tanh(x)
        t
    end

    function backward(lyr::TanhLayer, dout::AbstractArray{T}) where T
        dout .* (1 .- lyr.t .^ 2)
    end

# Dropout

mutable struct Dropout <: AbstractLayer
    ratio
    mask
    Dropout() = new()
end

function forward(self::Dropout, x, train=true)
    self.ratio = 0.5
    if train == true
        self.mask = rand(size(x)...) .> self.ratio
        return x .* self.mask
    else
        return x .* (1.0 - self.ratio)
    end
end

function backward(self::Dropout, dout)
    return dout .* self.mask
end

# Batch Normalization
mutable struct BatchNormalization <: AbstractLayer
gamma
beta
momentum
input_shape

running_mean
running_var

batch_size
xc
xn
std
dgamma
dbeta
function (::Type{BatchNormalization})(gamma, beta)
    self = new()
    self.gamma = gamma
    self.beta = beta
    self.momentum = 0.9
    self
end
end

function forward(self::BatchNormalization, x; train=true)
    self.input_shape = size(x)
    if length(size(x)) != 2
        C, H, W, N = size(x)
        x = reshape(x, C*H*W, N)
    end
    
    if !isdefined(BatchNormalization, :running_mean)
        D, N = size(x)
        self.running_mean = zeros(D)
        self.running_var = zeros(D)
    end
    
    if train == true
        mu = mean(x, dims=2)
        self.xc = xc = x .- mu
        var = mean(xc.^2, dims=2)
        self.std = std = sqrt.(var .+ 10e-7)
        self.xn = xn = xc ./ std
        
        self.batch_size = size(x, 2)
        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) .* mu
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) .* var
    else
        xc = x .- self.running_mean
        xn = xc ./ sqrt.(self.running_var + 10e-7)
    end
    out = self.gamma .* xn .+ self.beta
    return reshape(out, size(x))
end

function backward(self::BatchNormalization, dout)
    if length(size(dout)) != 2
        C, H, W, N = size(dout)
        dout = reshape(dout, C*H*W, N)
    end
    self.dbeta = dbeta = sum(dout, dims=2)
    self.dgamma = dgamma = sum(self.xn .* dout, dims=2)
    dxn = self.gamma .* dout
    dxc = dxn ./ self.std
    dstd = -sum((dxn .* self.xc) ./ (self.std .^ 2), dims=2)
    dvar = 0.5 .* dstd ./ self.std
    dxc .+= (2.0 / self.batch_size) .* self.xc .* dvar
    dmu = sum(dxc, dims=2)
    dx = dxc .- dmu ./ self.batch_size
    
    dx = reshape(dx, self.input_shape)
    return dx
end

# Loss function
    mutable struct BCELossLayer <: AbstractLayer
        loss
        y
        t
        BCELossLayer() = new()
    end

    function forward(lyr::BCELossLayer, y::AbstractArray, t::AbstractArray)
        lyr.y = y
        lyr.t = t
        loss = lyr.loss = Functional.BCELoss(y, t)
        return loss
    end

    function backward(lyr::BCELossLayer, dout)
        return (lyr.y .- lyr.t) ./ (lyr.y .* (1 .- lyr.y))
    end
    
    mutable struct SoftmaxWithLossLayer{T} <: AbstractLayer
        loss::T
        y::AbstractArray{T}
        t::AbstractArray{T}
        (::Type{SoftmaxWithLossLayer{T}})() where {T} = new{T}()
    end

    function forward(lyr::SoftmaxWithLossLayer{T}, x::AbstractArray{T}, t::AbstractArray{T}) where T
        lyr.t = t
        y = lyr.y = Functional.softmax(x)
        lyr.loss = Functional.crossEntropyLoss(y, t)
    end

    function backward(lyr::SoftmaxWithLossLayer{T}, dout::T=one(T)) where {T<:AbstractFloat}
        dout .* _swlvec(lyr.y, lyr.t)
    end

    @inline _swlvec(y::AbstractArray{T}, t::AbstractVector{T}) where {T<:AbstractFloat} = y .- t
    @inline _swlvec(y::AbstractArray{T}, t::AbstractMatrix{T}) where {T<:AbstractFloat} = (y .- t) / size(t)[2]
end
