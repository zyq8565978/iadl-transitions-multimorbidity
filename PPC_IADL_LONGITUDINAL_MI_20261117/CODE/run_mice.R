Sys.setlocale('LC_CTYPE','English_United States.utf8')
Sys.setenv(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
if (nzchar(Sys.getenv("PPC_IADL_R_LIB"))) .libPaths(c(Sys.getenv("PPC_IADL_R_LIB"), .libPaths()))
suppressPackageStartupMessages(library(mice))
suppressPackageStartupMessages(library(miceadds))
stopifnot(as.character(packageVersion('mice'))=='3.19.0',as.character(packageVersion('miceadds'))=='3.20.10',
          as.character(packageVersion('lme4'))=='1.1.35.5')
# miceadds 3.20-10 coerces unnamed scalar options into unnamed lists, then
# replaces them with defaults (including the internal "xplsfacs" return mode).
# Restore the documented scalar-option semantics in this process only.
# No package file or statistical routine is modified.
extract_original <- getFromNamespace('mice_imputation_extract_list_arguments','miceadds')
extract_scalar_compatible <- function(micearg,vname,miceargdefault) {
  if(is.atomic(micearg) && length(micearg)==1L && is.null(names(micearg)))return(micearg)
  extract_original(micearg,vname,miceargdefault)
}
assignInNamespace('mice_imputation_extract_list_arguments',extract_scalar_compatible,ns='miceadds')
stopifnot(identical(extract_scalar_compatible('xplsfacs','bmi','pmm'),'xplsfacs'),
          identical(extract_scalar_compatible(0,'bmi',20),0),
          identical(extract_scalar_compatible(list(bmi='pmm'),'bmi','continuous'),'pmm'))
# R's chol returns upper U with t(U) %*% U = Sigma. The package expression
# t(U %*% z) instead has covariance U %*% t(U). Correct the multiplication.
fixed_normal_draw <- function(mu,Sigma) mu + as.vector(crossprod(chol(Sigma),stats::rnorm(length(mu))))
assignInNamespace('mice_multilevel_draw_rnorm1',fixed_normal_draw,ns='miceadds')
unit_cov <- matrix(c(4,1.6,1.6,1),2,2)
stopifnot(max(abs(crossprod(chol(unit_cov))-unit_cov))<1e-12)
stopifnot(identical(names(fixed_normal_draw(c(a=0,b=0),unit_cov)),c('a','b')))
args <- commandArgs(TRUE)
src <- args[1]; mode <- args[2]; batch <- as.integer(args[3])
root <- normalizePath('PPC_IADL_LONGITUDINAL_MI_20261117',winslash='/')
stopifnot(jsonlite::read_json(file.path(root,'QA/EXACT_NESTED_VARIANCE_GATE.json'))$pass)
source(file.path(root,'CODE/fast_ranef.R'))
enable_exact_nested_variance()
if(mode=='full')source(file.path(root,'CODE/warm_lmer.R'))
data <- read.csv(file.path(root,'PRIVATE',paste0(src,'_mi_input.csv')),check.names=FALSE)
meta <- read.csv(file.path(root,'PRIVATE',paste0(src,'_node_meta.csv')))
targets <- c('smoking','alcohol','bmi')
seed <- 61118000+match(src,c('CHARLS','HRS','ELSA','SHARE'))*10000+batch*37+ifelse(mode=='validation',1000,0)
stem <- paste(src,mode,batch,'r4',sep='_')
cache <- file.path(root,'PRIVATE',paste0(stem,'.rds'))
set.seed(seed)
truth <- data[,targets]; held <- matrix(FALSE,nrow(data),3,dimnames=list(NULL,targets))
if(mode=='validation') {
  if(src=='CHARLS') {
    eligible <- which(meta$wave==2015 & !is.na(data$bmi))
    families <- unique(data$family_id[eligible])
    selected <- sample(families,ceiling(.2*length(families)))
    held[eligible[data$family_id[eligible] %in% selected],'bmi'] <- TRUE
  } else {
    v <- if(src=='SHARE') 'smoking' else 'bmi'
    w <- switch(src,HRS=2016,ELSA=4,SHARE=7)
    held[meta$wave==w & !is.na(data[[v]]),v] <- TRUE
  }
  for(v in targets) {
    if(sum(held[,v])==0) {
      obs <- which(!is.na(data[[v]])); held[sample(obs,ceiling(.1*length(obs))),v] <- TRUE
    }
    # Remove masked measurements when they recur as a later-node lag auxiliary.
    maskkey <- paste(data$person_id[held[,v]],meta$wave[held[,v]])
    leak <- paste(data$person_id,meta$lag_wave) %in% maskkey
    data[leak,paste0('lag_',v)] <- 0
    data[leak,paste0('lag_',v,'_unknown')] <- 1
    data[held[,v],v] <- NA_real_
  }
}
data$bmi_low <- ifelse(is.na(data$bmi),NA_real_,as.numeric(data$bmi<18.5))
data$bmi_over <- ifelse(is.na(data$bmi),NA_real_,as.numeric(data$bmi>=25 & data$bmi<30))
data$bmi_obese <- ifelse(is.na(data$bmi),NA_real_,as.numeric(data$bmi>=30))
method <- setNames(rep('',ncol(data)),names(data)); method[targets] <- 'ml.lmer'
method['bmi_low'] <- '~I(as.numeric(bmi < 18.5))'
method['bmi_over'] <- '~I(as.numeric(bmi >= 25 & bmi < 30))'
method['bmi_obese'] <- '~I(as.numeric(bmi >= 30))'
pred <- matrix(0,ncol(data),ncol(data),dimnames=list(names(data),names(data)))
pred[targets,] <- 1;diag(pred) <- 0
pred[,c('person_id','family_id')] <- 0
pred['bmi',c('bmi_low','bmi_over','bmi_obese')] <- 0
vl <- setNames(rep('',ncol(data)),names(data))
lev <- setNames(rep(list(c('person_id','family_id')),3),targets)
model <- setNames(as.list(rep('pmm',3)),targets)
maxit <- switch(mode,pilot=2L,profile=1L,validation=5L,full=10L)
if(mode=='full' && length(args)>=4)maxit <- as.integer(args[4])
stopifnot(mode!='full' || maxit %in% c(10L,20L,40L))
m <- switch(mode,pilot=1L,profile=1L,validation=5L,full=5L)
notes <- character();tick <- proc.time()[3]
if(mode=='profile')Rprof(file.path(root,'QA',paste0(stem,'_timing.out')))
extra_args <- list(
  levels_id=lev,variables_levels=vl,model=model,donors=5,draw.fixed=TRUE,
  aggregate_automatically=FALSE,iter_re=5,random.effects.shrinkage=1e-6,
  remove.collinear=FALSE,printFlag=TRUE,glmer.warnings=TRUE)
imp <- withCallingHandlers({
  if(file.exists(cache)) {
    saved <- readRDS(cache)
    stopifnot(identical(saved$data,data),saved$m==m)
    if(saved$iteration<maxit) {
      file.copy(cache,sub('.rds$',paste0('_it',saved$iteration,'.rds'),cache),overwrite=FALSE)
      do.call(mice::mice.mids,c(list(obj=saved,maxit=maxit-saved$iteration),extra_args))
    } else saved
  } else do.call(mice,c(list(data=data,m=m,maxit=maxit,seed=seed,method=method,
    predictorMatrix=pred,visitSequence=c('bmi','bmi_low','bmi_over','bmi_obese','smoking','alcohol')),extra_args))
},
  warning=function(w){notes <<- c(notes,conditionMessage(w));invokeRestart('muffleWarning')},
  message=function(w){notes <<- c(notes,conditionMessage(w));invokeRestart('muffleMessage')})
if(mode=='profile') {
  Rprof(NULL)
  pp <- summaryRprof(file.path(root,'QA',paste0(stem,'_timing.out')))
  write.csv(pp$by.self,file.path(root,'QA',paste0(stem,'_profile_self.csv')))
  write.csv(pp$by.total,file.path(root,'QA',paste0(stem,'_profile_total.csv')))
}
saveRDS(imp,cache,compress=TRUE)
elapsed <- proc.time()[3]-tick
diagrows <- list(); validrows <- list()
for(i in seq_len(m)) {
  co <- complete(imp,i)
  stopifnot(all(is.finite(as.matrix(co[,targets]))),all(co$smoking %in% c(0,1)),all(co$alcohol %in% c(0,1)),all(co$bmi>=10 & co$bmi<=80))
  for(v in targets) {
    known <- !is.na(data[[v]])
    stopifnot(identical(co[[v]][known],data[[v]][known]))
    miss <- !known
    diagrows[[length(diagrows)+1]] <- data.frame(source=src,mode=mode,batch=batch,imputation=i,variable=v,
      imputed=sum(miss),mean=mean(co[[v]][miss]),sd=sd(co[[v]][miss]),minimum=min(co[[v]][miss]),maximum=max(co[[v]][miss]))
    if(mode=='validation') {
      j <- held[,v]
      validrows[[length(validrows)+1]] <- data.frame(source=src,variable=v,imputation=i,n=sum(j),
         observed_mean=mean(truth[[v]][j]),imputed_mean=mean(co[[v]][j]),bias=mean(co[[v]][j]-truth[[v]][j]),
         RMSE=sqrt(mean((co[[v]][j]-truth[[v]][j])^2)))
    }
  }
  if(mode=='full') write.csv(co[,targets],file.path(root,'PRIVATE',sprintf('%s_mi_%03d.csv',src,(batch-1)*5+i)),row.names=FALSE)
}
write.csv(do.call(rbind,diagrows),file.path(root,'QA',paste0(stem,'_distributions.csv')),row.names=FALSE)
if(length(validrows))write.csv(do.call(rbind,validrows),file.path(root,'TABLES',paste0(stem,'_validation.csv')),row.names=FALSE)
write.csv(as.data.frame(table(notes)),file.path(root,'QA',paste0(stem,'_conditions.csv')),row.names=FALSE)
if(!is.null(imp$loggedEvents))write.csv(imp$loggedEvents,file.path(root,'QA',paste0(stem,'_logged_events.csv')),row.names=FALSE)
jsonlite::write_json(list(source=src,mode=mode,batch=batch,m=m,iterations=imp$iteration,seed=seed,elapsed_seconds=unname(elapsed),
  observed_values_unchanged=TRUE,legal_values=TRUE,complete=TRUE,notes=as.list(table(notes))),
  file.path(root,'QA',paste0(stem,'_status.json')),pretty=TRUE,auto_unbox=TRUE)
cat('COMPLETE',stem,'seconds',elapsed,'\n')
