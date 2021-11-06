#!/usr/bin/env Rscript

# devtools::install_github("D-iii-S/d3srutils", upgrade="never")

require("tidyverse")
require("Hmisc")
require("d3s")

# https://stackoverflow.com/a/37974262
weighted.ttest.ci <- function(x, weights, conf.level = 0.95) {
    nx <- length(x)
    if (nx < 5) {
        x.indices <- sample(1:nx, size=2000, replace=RUE)
        x <- x[x.indices]
        weights <- weights[x.indices]
        nx <- length(x)
    }
    df <- nx - 1
    vx <- wtd.var(x, weights, normwt = TRUE) ## From Hmisc
    mx <- weighted.mean(x, weights)
    stderr <- sqrt(vx/nx)
    tstat <- mx/stderr ## not mx - mu
    alpha <- 1 - conf.level
    cint <- qt(1 - alpha/2, df)
    cint <- tstat + c(-cint, cint)
    data.frame(list(
        "y"=mx,
        "ymin"=cint[1]*stderr,
        "ymax"=cint[2]*stderr
    ))
}

bootstrapped.ci <- function(x, weights, conf.level = 0.95) {
    
}

if (!("tidyverse" %in% names(sessionInfo()$otherPkgs))) {
    require("tidyverse")
    require("Hmisc")
}

argv <- commandArgs(TRUE)

stopifnot("Provide output CSV and input filename(s) as arguments." = length(argv) >= 2)

ratios <- NULL
for (arg in argv[2:length(argv)]) {
    logger_info("Loading from %s", arg)
    arg.csv <- read_csv(arg)
    if ("jdk_one" %in% names(arg.csv)) {
        arg.csv <- arg.csv %>%
            rename(jdk_old_long=jdk_one_long, jdk_new_long=jdk_two_long) %>%
            rename(jdk_old=jdk_one, jdk_new=jdk_two)
    }
    if (is.null(ratios)) {
        ratios <- arg.csv
    } else {
        ratios <- ratios %>% add_row(arg.csv)
    }
}


logger_info("Renaming columns and cleaning-up the input data...")
ratios <- ratios %>%
    #mutate(ratio=1/ratio) %>%
    unite(name, sep=" ", remove=FALSE, benchmark, provider, jdk_old, jdk_new)


if (FALSE) {
    logger_info("Loading CSV with change description...")
    changes <- read_csv("../no-cgroups/changes.csv")

    changes <- changes %>%
        add_row(changes %>% mutate(jdk_new_long = jdk_old_long, change_zero=0)) %>%
        add_row(changes %>% mutate(jdk_old_long = jdk_new_long, change_zero=0))

    x <- right_join(ratios, changes)
} else {
    x <- ratios
    x$change_zero <- 0
}

logger_info("Computing the CI intervals (t-test distribution based) ...")
x.ci <- x %>%
    #filter(weight > 1.5e8) %>%
    group_by(type, kind, name, benchmark, provider, jdk_old, jdk_old_long, jdk_new, jdk_new_long, change_zero) %>%
    filter(!is.na(provider)) %>%
    summarise(
        count=n(),
        ci=list(weighted.ttest.ci(ratio, weight) %>% rename(ci_mean=y, ci_lwr=ymin, ci_upr=ymax))
    ) %>%
    unnest(cols=c(ci)) %>%
    ungroup()


logger_info("Writing to %s", argv[1])
write_csv(x.ci, argv[1])
