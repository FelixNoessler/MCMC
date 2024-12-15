using Distributions
using CairoMakie
using LinearAlgebra

## generate two mvn distributions
d1 = MvNormal([1.0, 1.0], [1.0 0.5; 0.5 1.0])
d2 = MvNormal([1.0, 1.0], [1.0 0.5; 0.5 1.0])

## generate samples
n = 1000
x1 = rand(d1, n)
x2 = rand(d2, n)

## plot
let
    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1)
    scatter!(x1[1, :], x1[2, :], markersize = 3, color = :blue)
    scatter!(x2[1, :], x2[2, :], markersize = 3, color = :red)
    display(fig)
end

## kullback leibler divergence
function kl_divergence(d1, d2)
    μ_x = d1.μ
    Σ_x = d1.Σ
    μ_z = d2.μ
    Σ_z = d2.Σ

    k = length(μ_x)  # Dimensionality
    Σ_z_inv = inv(Σ_z)
    term1 = tr(Σ_z_inv * Σ_x)
    term2 = (μ_z - μ_x)' * Σ_z_inv * (μ_z - μ_x)
    term3 = log(det(Σ_z) / det(Σ_x))
    kl = 0.5 * (term1 + term2 - k + term3)
    return kl
end


kl_divergence(d1, d2)


## One dimensional case
kl_one_dim(d1, d2) = log(d2.σ / d1.σ) + (d1.σ^2 + (d1.μ - d2.μ)^2) / (2 * d2.σ^2) - 0.5

let
    d1 = Normal(0.0, 4.0)  # P
    d2 = Normal(-7, 2)     # Q

    fig = Figure()
    ax = Axis(fig[1, 1]; title = "amount of information being lost\n … by using Q to approximates P: KL(P||Q) = $(round(kl_one_dim(d1, d2); digits = 2))\n … by using P to approximates Q: KL(Q||P) = $(round(kl_one_dim(d2, d1); digits = 2))",
              ylabel = "Density", xlabel = "x")
    plot!(d1, color = :blue, label = "p(x)")
    plot!(d2, color = :red, label = "q(x)")
    axislegend()
    display(fig)
end


let
    μ_q = 0:0.01:20

    d1 = Normal(0.0, 1.0)               # P
    d2 = [Normal(m, 1.0) for m in μ_q]  # Q

    kl_p_q = kl_one_dim.(d1, d2)
    kl_q_p = kl_one_dim.(d2, d1)

    fig = Figure()
    Axis(fig[1,1], xlabel = "μ_q", ylabel = "KL", title = "μ_p = σ_p = σ_q = 1\nif both distributions have the same standard deviation: KL(P||Q) = KL(Q||P)")
    lines!(μ_q, kl_p_q, color = :blue, label = "KL(P||Q)")
    lines!(μ_q, kl_q_p, color = :red, label = "KL(Q||P)")
    axislegend()
    fig
end

let
    σ_q = 0.4:0.01:3

    d1 = Normal(0.0, 1.0)               # P
    d2 = [Normal(0.0, v) for v in σ_q]  # Q

    kl_p_q = kl_one_dim.(d1, d2)
    kl_q_p = kl_one_dim.(d2, d1)

    fig = Figure()
    Axis(fig[1,1], xlabel = "σ_q", ylabel = "KL", title = "μ_p = μ_q = 0, σ_p = 1\nif both distributions have different standard deviations: KL(P||Q) ≠ KL(Q||P)")
    lines!(σ_q, kl_p_q, color = :blue, label = "KL(P||Q)")
    lines!(σ_q, kl_q_p, color = :red, label = "KL(Q||P)")
    axislegend()
    fig
end
