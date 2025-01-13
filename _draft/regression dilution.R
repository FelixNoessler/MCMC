library(dplyr)
library(tidyr)
library(ggplot2)

df <- tibble(
    true_x = seq(0, 10, length.out = 100),
    measured_x = true_x + rnorm(100, sd = 5),
    y = 2 * true_x + rnorm(100, sd = 3)
)

pivot_longer(df, cols = c(measured_x, true_x),
             names_to = "x_type", values_to = "x") %>%
    ggplot(aes(x, y)) +
    geom_point() +
    geom_smooth(method = "lm") +
    facet_wrap(~x_type)

lm(y ~ measured_x, data = df)
