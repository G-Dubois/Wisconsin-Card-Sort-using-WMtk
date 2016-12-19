## TD-learning with HRRs WCST proof of concept 
## 
## Copyright 2016 Joshua L. Phillips
source("hrr.r")
require(Matrix)

## Set the seed for repeatability
set.seed(0)

## This is the number of dimensions to attend to
## which could be color, shape, size, etc.
ndims <-2
## This is the number of features used in each
## dimension. For the color dimension, this could
## be red, blue, green, etc.
nfeatures <- 3
## The total number of possible cards
ncards <- nfeatures^ndims
#print(sprintf("ncards: %d", ncards))

## Length of the HRRs
n <- 64

## Identity vector
hrr.i <- rep(0,n)
hrr.i[1] <- 1

## We encode each feature as a unique HRR
features <- replicate(ndims,replicate(nfeatures,hrr(n,normalized=TRUE)))
write(features,file="temp_hrrs.dat",ncolumns=n)
features[,,] <- scan("temp_hrrs_c.dat")

#for (x in seq(1,ndims)) {
#    for (y in seq(1,nfeatures)) {
#        print(sprintf("[%d %x] = ",y,x))
#        print(features[,y,x])
#    }
#}

## Internal states need to be unique even
## though they signal items in the
## outside world. This is to ensure that
## concepts stored in working memory are
## tagged appropriately to a concept in
## the outside world, but are not
## represented by the -exact- same vector
## internally. This is needed in order
## to discriminate between -remembering-
## a concept and that concept being
## currently -present- in the environment.
internal <- as(sample(n),"pMatrix")
wm_metas <- array(do.call(cbind,lapply(split(features,sapply(seq(1,ndims*nfeatures),rep,n)),"%*%",internal)),dim=c(n,nfeatures,ndims))

## Now that we have all of the possible
## variables included, we will pre-compute
## all of the cards for efficiency. Note
## that this is not needed, but makes things
## faster below...
## Outer convolution using Reduce to produce
## all cards (not used, but a cool trick).
## cards <- Reduce(oconvolve,
##                 lapply(split(features,
##                              array(sapply(seq(1,ndims),
##                                           rep,
##                                           length(features[,,1])),
##                                    dim=dim(features))),
##                        matrix,nrow=n),
##                 accumulate=TRUE)[[ndims]]

## We initialize the weight vector with small
## random values. Make this the last random
## initialization so that it can be commented
## out once the weights have been trained.
W <- rep(0,n)
W <- W + ((runif(n)*0.02)-0.01)
write(W,file="temp_weights.dat",ncolumns=1)
W <- scan("temp_weights_c.dat")
bias <- 0

## Optionally, we may want to start with an
## optimistic critic. This can easily be
## performed using SVD, but requires all
## states be enumerated. It might be possible
## to simplify this in some way by only
## selecting a subset of the states or
## an by using iterative approach.
## Needs to be updated for current code if one
## wants to use this strategy.
## cards.svd <- svd(t(matrix(cards,nrow=n)))
## W <- as.vector(cards.svd$v %*% diag(1/cards.svd$d) %*% t(cards.svd$u) %*%
##                    matrix(1,nstates*(nstates+1)))

## This vector will be updated on each step
## to reflect how the weights should change
## so that states[x] %*% W -> V[s].
## Basically, it only plays a significant
## role if lambda > 0.0 (below).
eligibility <- rep(0,n)

## Standard reward for non-goal states.
## Normally, this will be zero since there
## is no feedback at intermediate stages.
## However, other reward policies exist
## which will change the V[s] values but
## also speed convergence (like when
## setting to -1, for example).
default_reward <- 0.0

## The reward for a correct
## classification
correct_reward <- 1.0

## Discounted future rewards
## V[s] = r[s] + gamma*V[s+1]
## delta[s] = (r[s] + gamma*V[s+1]) - V[s]
gamma <- 0.5

## For stability of learning, we don't
## update the weight vector using the
## exact delta[s] values. Instead, we
## use a fraction (lrate) of that update.
lrate <- 0.9

## Eligibility traces
## Under TD[lambda] rules, we share some
## of the information from our current
## delta[s] with states in the recent
## past since they also contributed to
## our current position. This can easily
## be set to 0.0 for standard TD-learning,
## or it can be set higher to speed
## learning (although values too close to
## 1.0 will make the equations unstable
## unless the lrate is also lowered
## to compensate).
lambda <- 0.1

## Epsilon-soft policy
epsilon <- 0.05

## Epsilon-forced WM
epsilon_force <- 0.0

## Number correct in-a-row so far...
count <- 0
max_count <- 0

## Clear weight eligibility vector
eligibility <- rep(0,n)
bias_e <- 0.0

## This is the agent's current thinking
## regarding the world.
## Initialize with a default expectation
## where it has no real idea of how to
## perform.
current_wm <- NULL

## Get ready for learning step
previous <- NULL
previous_hrr <- rep(0,n)
previous_r <- 0.0
previous_wm <- NULL
previous_value <- 0.0
current_value <- 0.0

map <- function(x,y){y[,x[1],x[2]]}

## Mostly for utility
make_rep <- function(input,wm) {
    if (is.null(wm))
        return (mconvolve(apply(input,2,map,features)))
    else
        return (mconvolve(cbind(apply(input,2,map,features),
                                map(wm,wm_metas))))
}

# Set up the ruleset
rules <- read.table("temp_rules_c.dat",header=FALSE,sep=" ")+1
rulenum = 1

## Set up the current rule
#rule <- c(sample(nfeatures,1),sample(ndims,1))
rule <- c(rules[rulenum,1], rules[rulenum,2])
rulenum <- rulenum + 1
switches <- c(1,1)
print(sprintf("Rule change: [%d %d]",rule[1],rule[2]))

## Set the maximum number of presentations
## before we quit
nsteps <- 5

## Percepts
percepts <- replicate(nsteps,sample(nfeatures,ndims,replace=TRUE))
write(percepts-1,file="temp_percepts.dat",ncolumns=ndims)
percepts[,] <- scan("temp_percepts_c.dat")+1

debug <- FALSE

## This is a continouous learning task, so
## there is no absorbing reward state...
for (timestep in seq(1,nsteps)) {

    ## Always use debug?
    ## debug <- TRUE

    ## Get current percept
    current <- rbind(percepts[,timestep],seq(1,ndims))
    current_hrr <- mconvolve(apply(current,2,map,features))
    
    ## Internal update of working memory is done -before-
    ## performing an action. Note this involves a juggling
    ## of the possible WM options, and loading up the one with
    ## the largest value. The TD error is then computed -after-
    ## since this reasoning process is considered "instantaneous".
    possible_wm <- unique(cbind(current_wm,current),MARGIN=2)
    
    possible_values <- c(as.vector(W%*%current_hrr+bias),
                         apply(possible_wm,2,
                               function(x,y) {as.vector(W%*%convolve(map(x,wm_metas),y)+bias)},
                               current_hrr))
    wm_move <- which(possible_values==max(possible_values))[1]
    ## Epsilon-soft updates
    if (runif(1) < epsilon) {
        wm_move <- sample(length(possible_values),1)
        if (debug)
            print("Random WM")
    }
    ## Softmax WM updates
    ## wm_move <- which(runif(1)<cumsum(exp(possible_values/epsilon)/
    ##                                      sum(exp(possible_values/epsilon))))[1]

    ## Get information for TD-update
    if (wm_move == 1) {
        current_wm <- NULL
        wm_hrr <- hrr.i
    }
    else {
        current_wm <- possible_wm[,wm_move-1]
        wm_hrr <- map(current_wm,wm_metas)
    }
    current_value <- possible_values[wm_move]
    current_hrr <- convolve(current_hrr,wm_hrr)

    ## Forced WM with rule... causes perfect performance as expected...
    if (runif(1) < epsilon_force) {
        current_wm <- rule
        current_hrr <- make_rep(current,current_wm)
        current_value <- as.vector(W%*%current_hrr)+bias
    }

    if (debug)
        sapply(sprintf("[ %d %d | %d %d ]: %g",current[1,1],current[1,2],c(0,possible_wm[1,]),c(0,possible_wm[2,]),possible_values),print)
    
    if (percepts[rule[2],timestep]==rule[1])
        correct <- 1
    else
        correct <- 2

    if (debug)
        if (!is.null(current_wm))
            if ((current_wm[1] == rule[1]) & (current_wm[2] == rule[2]))
                print("Loaded correct rule...")


    ## Standard TD update
    td_error <- (previous_r + gamma*current_value) - previous_value
    if (debug)
        print(sprintf("%d: %g = (%g + %g*%g) - %g",
                      timestep,
                      td_error,
                      previous_r,gamma,current_value,previous_value))
    ## td_error <- previous_r - previous_value
    ## if (debug)
    ##     print(sprintf("%d: %g = %g - %g",
    ##                   timestep,
    ##                   td_error,
    ##                   previous_r,previous_value))
    eligibility <- (lambda*eligibility) + previous_hrr
    ## bias_e <- 1.0
    W <- W + lrate*eligibility*td_error
    bias <- bias + lrate*bias_e*td_error
    

    ## Time to make a move!
    if (is.null(current_wm)) {
        ## Make a random pile choice
        ## This is the prepotent strategy even if it
        ## differs a little from previous WCST network
        ## behaviours which would at least have some
        ## error-driven learning to help make the
        ## decision.
        move <- sample(2,1)
    }
    else {
        move <- 2
        if (percepts[current_wm[2],timestep]==current_wm[1])
            move <- 1
    }

    ## Were we correct?
    if (move == correct) {
        current_r <- correct_reward
        count <- count + 1
    }
    else {
        current_r <- default_reward
        count <- 0
    }

    if (is.null(current_wm)) {
        wm_printable <- c(0,0)
    }
    else {
        wm_printable <- current_wm
    }


    if (debug)
        print(sprintf("%d: percept [%d %d], rule [%d %d], wm [%d %d], move %d [%d] -> %d",
                      timestep,
                      current[1,1],current[1,2],
                      rule[1],rule[2],
                      wm_printable[1],wm_printable[2],
                      move,correct,count))

    ## Change the rule!!
    if (count > 100) {
        count <- 0
        #new_rule <- rule
        #while (all(new_rule == rule))
        #    new_rule <- c(sample(nfeatures,1),sample(ndims,1))
        #rule <- new_rule
        rule <- c(rules[rulenum,1], rules[rulenum,2])
	rulenum <- rulenum + 1
	switches <- c(switches,timestep)
        print(sprintf("Timestep: %d - new rule [%d %d]",timestep,rule[1],rule[2]))
    }

    ## Getting close, so print things out
    #if (count > 90)
        debug <- TRUE
    #else
    #    debug <- FALSE
    
    ## Store previous information...
    previous <- current
    previous_hrr <- current_hrr
    previous_r <- current_r
    previous_wm <- current_wm
    previous_value <- current_value

    if (!is.null(current_wm))
        if (all(current_wm == rule) & move != correct)
            print("Broken!")
            
    ## if (count > max_count)
    ##     print(sprintf("%d: %d",timestep,count))
    
    max_count <- max(max_count,count)
    ## if (count > 0)
    ##     debug <- TRUE
    ## else
    ##     debug <- FALSE

    if (timestep %% 1000 == 0) {
        delta_t <- switches[-1]-switches[-length(switches)]
        plot(delta_t,ylab=expression(Delta*t),xlab="swtich #")
        print(sprintf("Timestep: %d (median switch time: %g)",timestep,median(delta_t)))
    }
    
}

## Absorbing TD update
td_error <- previous_r - previous_value
## print(c("Standard update:",sprintf("[(%f + %f*%f) - %f] = %f",
##                                    previous_r,gamma,current_value,
##                                    previous_value,td_error)))
eligibility <- (lambda*eligibility) + previous_hrr
## bias_e <- 1.0
W <- W + lrate*eligibility*td_error
bias <- bias + lrate*bias_e*td_error
