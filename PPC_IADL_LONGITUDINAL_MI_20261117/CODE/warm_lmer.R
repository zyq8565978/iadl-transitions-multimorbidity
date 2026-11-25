# Reuse only optimizer starting values; every FCS conditional model is refitted.
# This does not reuse a variance estimate as if it were a completed new fit.
lmer_call_native <- getFromNamespace('mice_multilevel_doCall_suppressWarnings','miceadds')
lmer_starts <- new.env(parent=emptyenv())
warm_lmer_call <- function(what,args,warnings=TRUE) {
  v <- get0('vname',envir=parent.frame(),inherits=FALSE)
  cold <- args
  if(!is.null(v) && exists(v,envir=lmer_starts,inherits=FALSE))args$start <- list(theta=get(v,envir=lmer_starts))
  ans <- tryCatch(lmer_call_native(what,args,warnings),error=function(e) {
    if(is.null(args$start))stop(e)
    warning('Cached variance-component start failed; using native cold start')
    lmer_call_native(what,cold,warnings)
  })
  msgs <- ans@optinfo$conv$lme4$messages
  if(!is.null(args$start) && length(msgs) && any(grepl('failed|degenerate|unable',msgs,ignore.case=TRUE))) {
    cold_ans <- lmer_call_native(what,cold,warnings)
    if(as.numeric(logLik(cold_ans))>=as.numeric(logLik(ans))-1e-8)ans <- cold_ans
  }
  if(!is.null(v))assign(v,lme4::getME(ans,'theta'),envir=lmer_starts)
  ans
}
assignInNamespace('mice_multilevel_doCall_suppressWarnings',warm_lmer_call,ns='miceadds')
