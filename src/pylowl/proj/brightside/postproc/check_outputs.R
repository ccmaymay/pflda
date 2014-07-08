read.subtree.vec <- function(f) {
    identifier <- scan(f, 'c', 1, quiet=TRUE)
    v <- scan(f, nlines=1, quiet=TRUE)
    return(list(identifier=identifier, v=v))
}

min.Elogpi <- function(filename) {
    f <- file(filename)
    open(f)
    eof <- FALSE
    m <- Inf
    while (! eof) {
        r <- read.subtree.vec(f)
        m <- min(r$v, m)
        if (length(r$identifier) == 0) {
            eof <- TRUE
        }
    }
    close(f)
    return(m)
}

for (d in list.files()) {
    filename <- paste(d, 'subtree_Elogpi', sep='/')
    if (file.exists(filename)) {
        cat(filename, min.Elogpi(filename), '\n')
    }
}
