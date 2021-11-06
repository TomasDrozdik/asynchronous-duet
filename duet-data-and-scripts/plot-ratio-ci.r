#!/usr/bin/env Rscript

# devtools::install_github("D-iii-S/d3srutils", upgrade="never")

require("tidyverse")
require("Hmisc")
require("d3s")
require("ggplot2")

argv <- commandArgs(TRUE)

stopifnot("Provide input CSV and plot base filename (such as plot.XXXX.png) as arguments." = length(argv) == 2)
x.ci <- read_csv(argv[1])

xx <- x.ci %>%
    filter(!is.na(ci_lwr)) %>%
    group_by(name) %>% mutate(have_both=n()) %>% ungroup() %>%
    mutate(sameversions=(jdk_old_long == jdk_new_long)) %>%
    mutate(name2=ifelse(sameversions, sprintf("%s ***", name), name)) %>%
    unite(type_kind, sep="-", type, kind, remove=FALSE) %>%
    mutate(typecolor=ifelse(sameversions, sprintf("%s (%s) - same versions", type, kind), sprintf("%s (%s) - different versions", type, kind))) %>%
    filter(have_both <= 20)

make.plot <- function(filename, xx, plot_extras=NULL) {
    logger_info("Plotting for %s ...", filename)
    xx.plot <- ggplot(xx, aes(y=name2, color=typecolor)) +
        #scale_x_log10() +
        geom_errorbarh(mapping=aes(xmin=ci_lwr, xmax=ci_upr, height=0.4)) +
        geom_vline(xintercept=1.0, col="gray") +
        facet_grid(rows=vars(benchmark), scales="free", space="free_y")
    ggsave(filename=filename,
        plot=xx.plot + theme_light(base_size=10) + plot_extras,
        width=12,
        height=36,
        units="in",
        dpi=150
    )
    logger_info("  ... plot saved.")
}

make.plot(sub("XXXX", "all.zoomed", argv[2]), xx, coord_cartesian(xlim=c(0.2, 1.15)))
#make.plot(sub("XXXX", "all.zoomed", argv[2]), xx, coord_cartesian(xlim=c(-0.7, 0.7)))
make.plot(sub("XXXX", "all.full", argv[2]), xx)

