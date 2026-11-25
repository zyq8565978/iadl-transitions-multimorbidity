# Official REML working covariance for the V4 B1 Gaussian domains.
args <- commandArgs(trailingOnly = TRUE)
library(nlme)
d <- read.csv(file.path(args[1], "input.csv"))
d$person <- factor(d$person)
x <- names(d)[-c(1L, 2L)]
msgs <- character()
fit <- withCallingHandlers(
  lme(reformulate(x, response = "y", intercept = FALSE), random = ~1 | person,
      data = d, method = "REML", na.action = na.fail,
      control = lmeControl(msMaxIter = 100, apVar = FALSE)),
  warning = function(w) {
    msgs <<- c(msgs, conditionMessage(w))
    invokeRestart("muffleWarning")
  }
)
stopifnot(identical(names(fixef(fit)), x))
options(digits = 17)
write.csv(data.frame(beta = unname(fixef(fit))), file.path(args[1], "coef.csv"), row.names = FALSE)
write.csv(unname(vcov(fit)), file.path(args[1], "vcov.csv"), row.names = FALSE)
write.csv(data.frame(residual_var = sigma(fit)^2,
                     random_intercept_var = as.numeric(VarCorr(fit)[1L, "Variance"]),
                     nlme_version = as.character(packageVersion("nlme")),
                     warnings = paste(msgs, collapse = " | ")),
          file.path(args[1], "diagnostics.csv"), row.names = FALSE)
