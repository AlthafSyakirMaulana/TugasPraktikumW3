logistic_regression_newton <- function(X, y, max_iter = 100, tol = 1e-6) {
  X <- as.matrix(X)
  y <- as.vector(y)
  n <- nrow(X)
  p <- ncol(X)
  beta <- rep(0, p)
  
  sigmoid <- function(z) 1 / (1 + exp(-z))
  
  for (i in 1:max_iter) {
    eta <- X %*% beta
    mu <- sigmoid(eta)
    W <- diag(as.vector(mu * (1 - mu)), n, n)
    z <- eta + (y - mu) / (mu * (1 - mu) + 1e-8)  # avoid division by zero
    H <- t(X) %*% W %*% X
    score <- t(X) %*% (y - mu)
    
    delta <- solve(H, score)
    beta_new <- beta + delta
    
    if (sum(abs(beta_new - beta)) < tol) break
    beta <- beta_new
  }
  
  fit <- sigmoid(X %*% beta)
  return(list(method = "Newton-Raphson", beta = beta, fit = fit))
}

logistic_regression_iwls <- function(X, y, max_iter = 100, tol = 1e-6) {
  X <- as.matrix(X)
  y <- as.vector(y)
  n <- nrow(X)
  p <- ncol(X)
  beta <- rep(0, p)
  
  sigmoid <- function(z) 1 / (1 + exp(-z))
  
  for (i in 1:max_iter) {
    eta <- X %*% beta
    mu <- sigmoid(eta)
    W <- diag(as.vector(mu * (1 - mu)), n, n)
    z <- eta + (y - mu) / (mu * (1 - mu) + 1e-8)
    
    beta_new <- solve(t(X) %*% W %*% X) %*% (t(X) %*% W %*% z)
    
    if (sum(abs(beta_new - beta)) < tol) break
    beta <- beta_new
  }
  
  fit <- sigmoid(X %*% beta)
  return(list(method = "IWLS", beta = beta, fit = fit))
}

# COBA
set.seed(42)
X <- cbind(1, matrix(rnorm(100), ncol = 2))  # tambah intercept
y <- rbinom(50, 1, 0.5)

# Newton-Raphson
res1 <- logistic_regression_newton(X, y)
print(res1)

# IWLS
res2 <- logistic_regression_iwls(X, y)
print(res2)
