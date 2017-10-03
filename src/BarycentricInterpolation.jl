module BarycentricInterpolation

export InterpolationGrid, InterpolationPolynomial, evaluate
export add_point!, change_point!
export chebyshev, discrete_chebyshev
export roots

mutable struct InterpolationGrid{T<:Real}
    x::Array{T,1}
    w::Array{T,1}
    s::T
    function InterpolationGrid{T}(x::Array{T,1}) where T<:Real
        d = length(x)-1
        w = zeros(T, d+1)
        w[1] = one(T)
        s = (maximum(x)-minimum(x))/2
        for k=2:d+1
            for i=1:k-1
                w[i] *= s/(x[i]-x[k])
            end
            w[k] = prod([s/(x[k]-x[j]) for j=1:k-1])
        end
        new(copy(x), w, s)
    end
end
InterpolationGrid(x::Array{T,1}) where T<:Real = InterpolationGrid{T}(x)


mutable struct InterpolationPolynomial{T<:Real}
    g::InterpolationGrid{T}
    f::Array{T,1}
    function InterpolationPolynomial{T}(g::InterpolationGrid{T}, f::Array{T,1}) where T<:Real
        if length(f)!=length(g.x)
            error("wrong length of f")
        end
        new(g, copy(f))
    end
end
InterpolationPolynomial(g::InterpolationGrid{T}, f::Array{T,1}) where T<:Real = 
    InterpolationPolynomial{T}(g, f)
InterpolationPolynomial(x::Array{T,1}, f::Array{T,1}) where T<:Real = 
    InterpolationPolynomial{T}(InterpolationGrid(x), f)


function evaluate{T<:Real}(x::T, p::InterpolationPolynomial{T})
    for j=1:length(p.f)
        if (x==p.g.x[j])
            return p.f[j]
        end
    end
    return sum(p.g.w.*p.f./(x - p.g.x))/sum(p.g.w./(x - p.g.x))
end

(p::InterpolationPolynomial){T}(x::T) = evaluate(x,p)

function add_point!{T<:Real}(g::InterpolationGrid{T}, x::T)
    push!(g.x, x)
    push!(g.w, zero(T))
    k = length(x) 
    for i=1:k-1
        w[i] *= g.s/(g.x[i]-g.x[k])
    end
    w[k] = prod([g.s/(g.x[k]-g.x[j]) for j=1:k-1])
end

function add_point!{T<:Real}(p::InterpolationPolynomial{T}, x::T, f::T)
    add_point!(p.g, x)
    push!(p.f, f)
end

function change_point!{T<:Real}(g::InterpolationGrid{T}, index::Integer, x::T)
    if x != g.x[index]
        k = length(g.x) 
        for i=1:k
            if i!=index
                g.w[i] *= (g.x[i]-g.x[index])/(g.x[i]-x)
            end     
        end
        g.w[index] = prod([g.s/(x-g.x[j]) for j=1:k if j!=index])
        g.x[index] = x
    end
end

function change_point!{T<:Real}(p::InterpolationPolynomial{T}, index::Integer, x::T)
     change_point!(p.g, index, x)
end

function change_point!{T<:Real}(p::InterpolationPolynomial{T}, index::Integer, x::T, f::T)
     change_point!(p.g, index, x)
     p.f[index] = f
end

chebyshev(d::Int, range::Tuple{Real, Real} = (-1.0,1.0)) = 
    ( d==0?InterpolationPolynomial([0.0],[1.0]):
     InterpolationPolynomial(range[1]+reverse(cos.((0:d)*pi/d)+1.0)*(range[2]-range[1])/2, (-1.0).^(d:-1:0)) )

function get_discrete_chebyshev_approx_nodes(n::Int, d::Int)
    m = ceil(Int,d/2)
    x = round.(Int,n*cos.((0:d)*pi/d)[m:-1:1])
    y=reverse(n-x)
    k0=length(y)
    for k=length(y):-1:1
        if y[k]<=k-1
            k0=k
            break
        end
    end
    y[1:k0] = 0:k0-1
    x=reverse(n-y)
    if isodd(d)
        return vcat(-reverse(x),x)
    else
        return vcat(-reverse(x),0,x)
    end
end


function discrete_chebyshev(n::Int, d::Int)
    x = get_discrete_chebyshev_approx_nodes(n,d)+0.0
    ll = length(x)
    f = -(-1.0).^(length(x):-1:1)
    p = InterpolationPolynomial(x, f)
    k = -n
    while true
        x = k+0.0
        y = p(x)
        if abs(y) > 1.0+1000*eps(Float64)
            #l1 = findfirst(p.g.x, x+1)
            #l2 = findfirst(p.g.x, x-1)
            l1 = findlast(t->t<x, p.g.x)
            l2 = findfirst(t->t>x, p.g.x)
            if l1>0 && sign(p.f[l1])==sign(y)
                change_point!(p, l1, x)
                change_point!(p, ll-l1+1, -x)
            elseif l2>0 && sign(p.f[l2])==sign(y)
                change_point!(p, l2, x)   
                change_point!(p, ll-l2+1, -x)
            else
                error("something strange happened")
            end
            k = -n
        else
            k += 1
        end
        if k==0
            break
        end
    end
    p
end


function discrete_chebyshev(X::Array{Float64,1}, d::Int)
    X = sort(X)
    n = length(X)
    m = ceil(Int, (d+1)/2)
    nn = vcat(1:m,n-(d+1-m)+1:n)
    ff = -(-1.0).^(length(nn):-1:1)
    p = InterpolationPolynomial(X[nn],ff)
    while true
        y = p.(X)
        k = findfirst(x->abs(x)>1+1000*eps(Float64), y)
        if k==0
            break
        end
        l1 = findlast(x->x<X[k], p.g.x)
        l2 = findfirst(x->x>X[k], p.g.x)
        if l1>0 && sign(p.f[l1])==sign(y[k])
            change_point!(p, l1, X[k])
        elseif l2>0 && sign(p.f[l2])==sign(y[k])
            change_point!(p, l2, X[k])   
        else
            error("something strange happened")
        end    
    end
    p
end

function roots(p::InterpolationPolynomial{Float64})
    # This function implements a method given in 
    # Piers W. Lawrence, Robert M. Corless: Stability of rootfinding for 
    # barycentric Lagrange interpolants, Numer. Algor. (2014) 65: 447. 
    # https://doi.org/10.1007/s11075-013-9770-3
    n = length(p.f)-1
    A = diagm(vcat(0.0,p.g.x))
    A[2:end,1] = p.g.w
    A[1,2:end] = p.f
    S = diagm(vcat(1.0, [p.f[j]==0.0?1.0:sqrt(abs(p.g.w[j]/p.f[j])) for j=1:n+1]))
    A = S\A*S
    S1 = diagm(vcat(1.0/norm(A[:,1]),ones(n+1)))
    S2 = diagm(vcat(1.0/norm(A[1,:]),ones(n+1)))
    A = S1*A*S2
    B = diagm(vcat(0.0, ones(n+1)))
    r = sort(eigvals(A,B))
    if abs(r[1])==Inf && abs(r[2])==Inf
        return r[3:end]
    elseif abs(r[1])==Inf && abs(r[end])==Inf
        return r[2:end-1]
    elseif abs(r[end-1])==Inf && abs(r[end])==Inf    
        return r[1:end-2]
    else
        error("something strange happened")
    end
end


end
