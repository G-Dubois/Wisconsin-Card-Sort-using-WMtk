## HRR Utility Functions
## Copyright 2016 Joshua L. Phillips

## Make an HRR
hrr <- function(length,normalized=FALSE) {
    if (normalized) {
        myhrr <- runif((length-1) %/% 2, -pi, pi)
        if (length %% 2) {
            myhrr <- Re(fft(complex(modulus=1,argument=c(0,myhrr,-rev(myhrr))),inverse=TRUE))/length
        }
        else {
            myhrr <- Re(fft(complex(modulus=1,argument=c(0,myhrr,0,-rev(myhrr))),inverse=TRUE))/length
        }
    }
    else {
        myhrr <- rnorm(length,0,1.0/sqrt(length))
    }
    return (myhrr)
}

## Exact inverse
inv <- function(x) {
    myhrr <- fft(x)
    return(Re(fft(complex(modulus=1.0/Mod(myhrr),argument=-Arg(myhrr)),inverse=TRUE))/length(x))
}

## Approximate inverse (same as exact inverse for unitary HRRs)
pinv <- function(x) {
    ## return (c(x[1],rev(x[-1])))
    return (Re(fft(Conj(fft(x)),inverse=TRUE))/length(x))
}

## Convolution - the default R function is incorrect
convolve <- function(x,y,normalize=TRUE) {
    return (Re(fft(fft(x)*fft(y),inverse=TRUE))/length(x))
}

## Convolve a matrix of HRRs (one per column)
mconvolve <- function(x) {
    Re(fft(apply(apply(x,2,fft),1,prod),inverse=TRUE))/nrow(x)
}

## Outer-product-like convolution, where mconvolve is more
## like the inner product among a set of HRRs, this function
## does the outer-product expression between two matrices of
## HRRs.
oconvolve <- function(x,y) {
    x <- as.matrix(x)
    y <- as.matrix(y)
    matrix(apply(x,2,function(x,y)apply(y,2,convolve,x),y),
           ncol=ncol(x)*ncol(y))
}

## For clarity of coding
correlate <- function(x,y,invf=inv) {
    return (convolve(x,invf(y)))
}

## Composition
compose <- function(x,y) {
    return(Re(fft(complex(modulus=1,argument=Arg(fft(x)+fft(y))),inverse=TRUE))/length(x))
}

## Decomposition
decompose <- function(x,y) {
    return(compose(x,-y))
    ## For reference
    ## return(Re(fft(complex(modulus=1,argument=Arg(fft(x)-fft(y))),inverse=TRUE)/length(x)))
}

## Normalized dot product
dot <- function(x,y) {
    x <- as.vector(x)
    y <- as.vector(y)
    return (as.double(x%*%y)
            / (sqrt(sum(x^2))*sqrt(sum(y^2))))
}

## Normalized vector
norm <- function(x) {
    x / sqrt(sum(x^2))
}

## Power
pow <- function(x,k) {
    return(Re(fft(fft(x)^k,inverse=TRUE))/length(x))
}

## Scaled dot product
sdot <- function(x,y,s=1.0) {
    x <- fft(x)
    y <- fft(y)
    sdiff <- ((Arg(x)-Arg(y))%%(2*pi))-pi
    mean(Mod(x)*Mod(y)*cos( (((abs(sdiff)^s) * pi/(pi^s)) * sign(sdiff) ) + pi ))
}

## Code for testing
## set.seed(0)
## mysize <- 5
## myzero <- hrr(mysize)
## myone <- hrr(mysize)

## i1 <- hrr(mysize)
## i2 <- hrr(mysize)

## w <- hrr(mysize)

## ip1 <- norm(convolve(i1,myzero)+convolve(i2,myzero))
## ip2 <- norm(convolve(i1,myzero)+convolve(i2,myone))
## ip3 <- norm(convolve(i1,myone)+convolve(i2,myzero))
## ip4 <- norm(convolve(i1,myone)+convolve(i2,myone))

## op1 <- myzero
## op2 <- myone
## op3 <- myone
## op4 <- myzero

## ips <- list(ip1,ip2,ip3,ip4)
## ops <- list(op1,op2,op3,op4)
## nops <- list(myone,myzero,myzero,myone)

## for (x in seq(1,4)) {
##     print("Pattern")
##     print(x)
##     print("Input")
##     print(ips[[x]])
##     print("Ouput")
##     print(convolve(ips[[x]],w))
##     print("Target")
##     print(ops[[x]])
## }

## mycomp <- rep(0,mysize)
## print("Invertables")
## for (x in seq(1,4)) {
##     temp <- inv(convolve(ips[[x]],inv(ops[[x]])))
##     mycomp <- mycomp + fft(temp)

##     temp <- convolve(ips[[x]],inv(nops[[x]]))
##     ## mycomp <- mycomp + fft(temp)
## }
## mycomp <- Re(fft(complex(modulus=1,argument=Arg(mycomp)),inverse=TRUE))/mysize

## print("Composition Check")
## for (x in seq(1,4)) {
##     print(c(convolve(ips[[x]],mycomp) %*% myzero,convolve(ips[[x]],mycomp) %*% myone))
## }

