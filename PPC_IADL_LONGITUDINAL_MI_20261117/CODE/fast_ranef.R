# Exact posterior variance for nested person/family random-intercept LMMs.
# No approximation, changed likelihood, new prior or altered random draw.
ranef_native <- getS3method('ranef','merMod',envir=asNamespace('lme4'))
ranef_nested_exact <- function(object,condVar=TRUE,drop=FALSE,whichel=c('person_id','family_id'),...) {
  fl <- lme4::getME(object,'flist')
  applicable <- setequal(names(fl),c('person_id','family_id')) &&
    all(lengths(object@cnms)==1L) && all(unlist(object@cnms)=='(Intercept)') &&
    !inherits(object,'glmerMod') && (is.null(weights(object)) || all(weights(object)==1))
  if(!applicable || !condVar || drop)return(ranef_native(object,condVar=condVar,drop=drop,whichel=whichel,...))
  pi <- as.integer(fl$person_id);fi <- as.integer(fl$family_id)
  map <- tapply(fi,pi,function(v)if(length(unique(v))==1L)v[1] else NA_integer_)
  if(anyNA(map))return(ranef_native(object,condVar=condVar,drop=drop,whichel=whichel,...))
  n <- tabulate(pi,nbins=nlevels(fl$person_id));nf <- nlevels(fl$family_id)
  vv <- lme4::VarCorr(object);vp <- as.numeric(vv$person_id);vf <- as.numeric(vv$family_id);ve <- sigma(object)^2
  den <- ve+n*vp
  effective <- as.numeric(rowsum(n/den,factor(map,levels=seq_len(nf)),reorder=TRUE))
  vfpost <- vf/(1+vf*effective)
  vppost <- vp*ve/den+(n*vp/den)^2*vfpost[map]
  ans <- ranef_native(object,condVar=FALSE,drop=FALSE,whichel=whichel,...)
  post <- list(person_id=vppost,family_id=vfpost)
  for(v in names(ans))attr(ans[[v]],'postVar') <- array(post[[v]],c(1,1,length(post[[v]])))
  ans
}
enable_exact_nested_variance <- function()registerS3method('ranef','merMod',ranef_nested_exact,envir=asNamespace('lme4'))
