read.subtree.vec <- function(f) {
    doc.id <- scan(f, 'c', 1, quiet=TRUE)
    v <- scan(f, nlines=1, quiet=TRUE)
    return(list(doc.id=doc.id, v=v))
}

min.subtree.elt <- function(filename) {
    f <- file(filename)
    open(f)
    eof <- FALSE
    m <- Inf
    while (! eof) {
        r <- read.subtree.vec(f)
        m <- min(r$v, m)
        if (length(r$doc.id) == 0) {
            eof <- TRUE
        }
    }
    close(f)
    return(m)
}

for (d in list.files()) {
    filename <- paste(d, 'subtree_Elogpi', sep='/')
    if (file.exists(filename)) {
        cat(filename, min.subtree.elt(filename), '\n')
    }
}
