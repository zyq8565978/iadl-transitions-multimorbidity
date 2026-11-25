Sys.setlocale('LC_CTYPE','English_United States.utf8')
txt <- paste(readLines('PPC_IADL_LONGITUDINAL_MI_20261117/CODE/diagnose_mi.R',warn=FALSE),collapse='\n')
txt <- gsub("normalizePath('PPC_IADL_LONGITUDINAL_MI_20261117',winslash='/')",
            "normalizePath('PPC_IADL_ALL_METHODS_MI_20261118',winslash='/')",txt,fixed=TRUE)
eval(parse(text=txt),envir=globalenv())
