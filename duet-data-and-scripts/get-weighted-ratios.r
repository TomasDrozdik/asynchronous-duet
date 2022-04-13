#!/usr/bin/env Rscript

get.overlap.length <- function(start.a, end.a, start.b, end.b) {
    min(c(end.a, end.b)) - max(c(start.a, start.b))
}

get.empty.output <- function() {
    tibble(
        benchmark=character(),
        run_id=character(),
        weight=numeric(),
        ratio=numeric(),
        jdk_one=character(),
        jdk_one_long=character(),
        jdk_two=character(),
        jdk_two_long=character(),
        provider=character(),
        type=character(),
        kind=character()
   )
}

compute.one.fragment <- function(x.ones, x.twos) {
    ratios <- get.empty.output()

    x.nrow <- nrow(x.ones)
    for (i in 1:x.nrow) {
        i.row <- x.ones[i,]
        i.start <- i.row$start_time[1]
        i.end <- i.row$end_time[1]
        i.run_id <- i.row$run_id[1]

        # We can safely assume that provider is part of run_id
        others <- x.twos %>%
            filter(benchmark == i.row$benchmark[1]) %>%
            filter(run_id == i.row$run_id[1]) %>%
            filter((start_time >= i.start & start_time <= i.end) | (i.start >= start_time & i.start <= end_time))

        if (nrow(others) > 0) {
            i.weights <- c()
            i.ratios <- c()

            for (j in 1:nrow(others)) {
                j.row <- others[j,]
                ij.weight <- get.overlap.length(i.start, i.end, j.row$start_time, j.row$end_time)
                ij.ratio <- log((i.end - i.start) / (j.row$end_time - j.row$start_time))

                i.weights <- c(i.weights, ij.weight)
                i.ratios <- c(i.ratios, ij.ratio)
            }

            ratios <- ratios %>%
                add_row(
                    weight=i.weights,
                    ratio=i.ratios,
                    benchmark=i.row$benchmark[1],
                    run_id=i.row$run_id[1],
                    jdk_one=i.row$jdk_version[1],
                    jdk_one_long=i.row$jdk[1],
                    jdk_two=j.row$jdk_version[1],
                    jdk_two_long=j.row$jdk[1],
                    provider=i.row$provider[1],
                    type="pair",
                    kind=i.row$kind[1],
                )
        }
    }

    ratios
}



argv <- commandArgs(TRUE)
stopifnot("Provide input and output CSV as a parameters"= length(argv) %in% c(2))

# devtools::install_github("D-iii-S/d3srutils", upgrade="never")

require("tidyverse")
require("Hmisc")
require("d3s")


x <- read_csv(argv[1])
x$kind <- "duet"
#x <- x %>% filter(benchmark == "lusearch")

if (!("run_id" %in% names(x))) {
    x$run_id <- paste(sep="--", x$benchmark, x$machine, x$provider, x$time)
}

if ("epoch_start_ms" %in% names(x)) {
    x$start_time <- x$epoch_start_ms
    x$end_time <- x$start_time + x$iteration_time_ns / 1000 / 1000
}

source("parallel.r")

ratios <- get.empty.output()

x <- x %>% unite(fragment_id, sep="--", benchmark, provider, kind, remove=FALSE)
x$jdk_version <- as.character(x$jdk_version)

x.ones <- x %>% filter(pair == "one")
x.twos <- x %>% filter(pair == "two")

x.fragments <- unique(x$fragment_id)

if (FALSE) {
    logger_progress_init("Computing ratios, %5.1f%% of fragments done.", length(x.fragments))
    for (i in x.fragments) {
        new_row <- compute.one.fragment(
            x.ones %>% filter(fragment_id == i),
            x.twos  %>% filter(fragment_id == i))
        ratios <- ratios %>% add_row(.data=new_row)
        logger_progress()
    }
} else {
    jobs.init()
    logger_progress_init("Computing ratios, %5.1f%% of fragments submitted.", length(x.fragments), 0.05)
    for (i in x.fragments) {
        jobs.submit(compute.one.fragment(
            x.ones %>% filter(fragment_id == i),
            x.twos %>% filter(fragment_id == i)
        ))
        logger_progress()
    }
    logger_info(" .. waiting for jobs to finish.")
    results <- jobs.collect()
    for (i in names(results)) {
        ratios <- ratios %>% add_row(.data=results[[i]])
    }
}

logger_info("Computation finished.")

write_csv(ratios, argv[2])
