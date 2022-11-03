
# # Gibbs sampling

# **General procedure:**
# - randomly order all the parameters $ θ_1 ... θ_d$
# - sample from $θ_j$ conditional distribution given all the other parameters of $\theta$:
# $$ \begin{align} 
#           &p(θ_j | θ^{t-1}_{-j}, y) \\
#           θ^{t-1}_{-j} &= (θ_1^t, ..., θ^t_{j-1}, θ^{t-1}_{j-1}, ..., θ^{t-1}_{d}  ) \\
# \end{align}$$
#
# **Example: Bivariate normal distribution**

using Distributions, Statistics, CairoMakie

own_theme = Theme(
    Axis=(xgridvisible=false, ygridvisible=false,
          topspinevisible=false, rightspinevisible=false),
)
set_theme!(own_theme)
σ_1,σ_2, ρ = 1, 1, 0.4

d1 = MvNormal([20,100], [σ_1^2 σ_1*σ_2*ρ; σ_1*σ_2*ρ σ_2^2])

y = rand(d1, 5)

d2 = Normal()

x = -3:0.1:3
## lines(x, pdf.(d2, collect(x)))

scatter(y[1, :], y[2, :])


## Literate.markdown("gibbs_sampling.jl"; flavor = Literate.CommonMarkFlavor())

lines(x, pdf.(d2, collect(x)))