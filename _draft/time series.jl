using Distributions
using CairoMakie

set_theme!(theme_minimal())

function ar(ϕ, tend)
    d = Normal(0.0, 1)
    x = zeros(tend)

    for i in 2:tend
        ar_part = 0

        for p in eachindex(ϕ)
            if i-p < 1
                continue
            end
            ar_part += ϕ[p] * x[i-p]
        end

        x[i] = ar_part + rand(d)
    end

    return x
end


function ma(θ, tend)
    d = Normal(0.0, 1)
    x = zeros(tend)
    ϵ = zeros(tend)

    for i in 2:tend
        ma_part = 0

        for q in eachindex(θ)
            if i-q < 1
                continue
            end
            ma_part += θ[q] * ϵ[i-q]
        end
        ϵ[i] = rand(d)
        x[i] = ma_part + ϵ[i]
    end

    return x
end


function arma(θ, ϕ, tend)
    d = Normal(0.0, 1)
    x = zeros(tend)
    ϵ = zeros(tend)

    for i in 2:tend
        ma_part = 0
        ar_part = 0

        for q in eachindex(θ)
            if i-q < 1
                continue
            end

            ma_part += θ[q] * ϵ[i-q]
            ar_part += ϕ[q] * x[i-q]
        end

        ϵ[i] = rand(d)
        x[i] = ma_part + ar_part + ϵ[i]
    end

    return x
end


function autocorrelation(x)
    ncorr = 10
    r = Array{Float64}(undef, ncorr)

    μ = sum(x) / length(x)
    T = length(x)

    denominator = sum( (x .- μ) .^ 2 )

    for k in eachindex(r)
        numerator = 0.0

        for t in k+1:T
            numerator += (x[t] - μ) * (x[t-k] - μ)
        end

        r[k] = numerator / denominator

    end

    return r
end

function partial_autocorrelation(x)

end


let
    # x = ar([1], 500)
    # x = ma(repeat([1.0], 500), 500)
    x = arma([0, 0, 0, 0], [0.34, 0.0, 0.0, 0.0], 10000)
    display(lines(x))

    # xt, xt_1 = x[2:end], x[1:end-1]
    # scatter(xt, xt_1;
    #     color=(:steelblue, 0.8))
    fig, _, _ = scatter(autocorrelation(x))
    ylims!(-1, 1)
    fig
end


autocorrelation([10, 9, 11, 12])
