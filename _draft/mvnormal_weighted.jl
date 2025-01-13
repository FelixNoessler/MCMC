using Distributions
using CairoMakie

# Generate some data
n = 1000
p = 2

μ = [0.0, 0.0]
Σ = [1.0 0.5; 0.5 1.0]

rand(MvNormal(μ, Σ), n)
