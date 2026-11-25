# Execute the validated imputer in the new output root. No historical file edits.
Sys.setlocale('LC_CTYPE','English_United States.utf8')
old <- normalizePath('PPC_IADL_LONGITUDINAL_MI_20261117',winslash='/')
new <- normalizePath('PPC_IADL_ALL_METHODS_MI_20261118',winslash='/')
txt <- paste(readLines(file.path(old,'CODE/run_mice.R'),warn=FALSE),collapse='\n')
txt <- gsub("normalizePath('PPC_IADL_LONGITUDINAL_MI_20261117',winslash='/')",
            "normalizePath('PPC_IADL_ALL_METHODS_MI_20261118',winslash='/')",txt,fixed=TRUE)
txt <- gsub("file.path(root,'QA/EXACT_NESTED_VARIANCE_GATE.json')",
            "file.path(old,'QA/EXACT_NESTED_VARIANCE_GATE.json')",txt,fixed=TRUE)
txt <- gsub("file.path(root,'CODE/fast_ranef.R')","file.path(old,'CODE/fast_ranef.R')",txt,fixed=TRUE)
txt <- gsub("file.path(root,'CODE/warm_lmer.R')","file.path(old,'CODE/warm_lmer.R')",txt,fixed=TRUE)
txt <- gsub('seed <- 61118000','seed <- 71118000',txt,fixed=TRUE)
txt <- gsub("paste0('lag_',v,'_unknown')","paste0('lag_',v,'_unavailable')",txt,fixed=TRUE)
eval(parse(text=txt),envir=globalenv())
