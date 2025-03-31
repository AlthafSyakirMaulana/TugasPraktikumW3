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

# FUNGSI GABUNGAN

logistic_regression <- function(X, y, method = "newton", max_iter = 100, tol = 1e-6) {
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
    z <- eta + (y - mu) / (mu * (1 - mu) + 1e-8)  # stabilisasi
    
    if (tolower(method) == "newton") {
      score <- t(X) %*% (y - mu)
      H <- t(X) %*% W %*% X
      delta <- solve(H, score)
      beta_new <- beta + delta
    } else if (tolower(method) == "iwls") {
      beta_new <- solve(t(X) %*% W %*% X) %*% (t(X) %*% W %*% z)
    } else {
      stop("Method harus 'newton' atau 'iwls'")
    }
    
    if (sum(abs(beta_new - beta)) < tol) break
    beta <- beta_new
  }
  
  fit <- sigmoid(X %*% beta)
  return(list(method = method, beta = beta, fit = fit))
}

# output

# Contoh data
set.seed(42)
X <- cbind(1, matrix(rnorm(100), ncol = 2))  # dengan intercept
y <- rbinom(50, 1, 0.5)

# Gunakan Newton-Raphson
res_newton <- logistic_regression(X, y, method = "newton")
print(res_newton)

# Gunakan IWLS
res_iwls <- logistic_regression(X, y, method = "iwls")
print(res_iwls)

