require("parallel")

CORES_SLACK <- 1.5

.JOBS.CONFIG <<- list (
        tasks = 0,
        max = 1
)

jobs.init <- function (max.jobs = NULL) {
    if (is.null (max.jobs)) max.jobs <- detectCores () * CORES_SLACK
    .JOBS.CONFIG <<- list (
        tasks = 0,
        ids = list (),
        max = max.jobs,
        results = list ()
    )
}


.jobs.collect <- function () {

    x.done <- mccollect (wait = FALSE, timeout = 1)

    if (is.null (x.done)) return (0)

    for (i in names (x.done)) {
        ii <- sprintf ("pid-%s", i)

        if (!(ii %in% names (.JOBS.CONFIG$ids))) next
        if (ii %in% names (.JOBS.CONFIG$results)) next

        .JOBS.CONFIG$results [[ii]] <<- x.done [[i]]
        .JOBS.CONFIG$tasks <<- .JOBS.CONFIG$tasks - 1
    }

    return (length (x.done))
}


jobs.submit <- function (expr, name = NULL) {

    # Block to avoid excessive parallelism.
    while (.JOBS.CONFIG$tasks >= .JOBS.CONFIG$max) .jobs.collect ()

    x <- mcparallel (expr)

    if (is.null (name)) name <- sprintf('.job.%d', x$pid)
    x$user.job.name <- name

    .JOBS.CONFIG$ids [[ sprintf ('pid-%d', x$pid) ]] <<- x
    .JOBS.CONFIG$tasks <<- .JOBS.CONFIG$tasks + 1

    name
}


jobs.collect <- function () {
    while (.JOBS.CONFIG$tasks > 0) {
        # .jobs.collect sleeps for up to one second
        x <- .jobs.collect ()
    }

    res <- list ()

    for (x.pid in names (.JOBS.CONFIG$ids)) {
        x.job <- .JOBS.CONFIG$ids [[x.pid]]
        val <- .JOBS.CONFIG$results [[x.pid]]
        res [[x.job$user.job.name]] <- val
    }

    .JOBS.CONFIG$results <<- list ()
    .JOBS.CONFIG$ids <<- list ()

    return (res)
}

