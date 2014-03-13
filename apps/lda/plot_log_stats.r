library(ggplot2)
library(gridExtra)
library(data.table)

options(warn=1)

# The following function downloaded on 2014-03-12 from
# https://github.com/hadley/ggplot2/wiki/Share-a-legend-between-two-ggplot2-graphs
g_legend <- function(a.gplot) {
    tmp <- ggplot_gtable(ggplot_build(a.gplot))
    leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
    legend <- tmp$grobs[[leg]]
    return(legend)
}

plot.smooth <- function(d, dataset.name) {
    return(ggplot(aes(x=iter, y=val, group=experiment, color=experiment, fill=experiment), data=d) +
        stat_summary(fun.data=mean_sdl, geom='smooth', mult=1, size=1) +
        (if (!all(is.na(d$val.eb))) {
                stat_summary(aes(x=iter, y=val.eb), fun.data=mean_sdl, geom='errorbar', mult=1, size=1)
            } else {
                NULL
            }) +
        theme_bw() +
        ggtitle(dataset.name) +
        ylab('NMI (mean +/- stdev)') +
        xlab('document'))
}

plot.tng3.smooth <- function(d) {
    p.diff3 <- plot.smooth(subset(d, dataset == 'diff3'), 'diff3')
    p.rel3 <- plot.smooth(subset(d, dataset == 'rel3'), 'rel3')
    p.sim3 <- plot.smooth(subset(d, dataset == 'sim3'), 'sim3')

    legend <- g_legend(p.diff3)
    lheight <- sum(legend$height)

    grid.arrange(
        p.diff3 + theme(legend.position='none'),
        p.rel3 + theme(legend.position='none'),
        p.sim3 + theme(legend.position='none'),
        legend,
        nrow=2)
}

plot.experiments <- function(experiment.group.name, dataset.names, experiment.names, experiment.names.legend, stat.names, stat.names.legend) {
    if (is.null(stat.names.legend)) {
        stat.names.legend <- stat.names
    }
    if (is.null(experiment.names.legend)) {
        experiment.names.legend <- experiment.names
    }

    d.list <- list()

    cat(experiment.group.name, '\n')
    for (dataset.name in dataset.names) {
        cat('*', dataset.name, '\n')

        for (k in 1:length(stat.names)) {
            stat.name <- stat.names[k]
            stat.name.legend <- stat.names.legend[k]
            for (i in 1:length(experiment.names)) {
                experiment.name <- experiment.names[i]
                experiment.name.legend <- experiment.names.legend[i]
                cat('  -', experiment.name, stat.name, '\n')
                filename.in <- paste(experiment.name, '/',
                    dataset.name, '_', stat.name, '.tab', sep='')
                data.raw <- tryCatch(read.table(filename.in, header=T),
                    error=function(ex) {NULL})
                if (! is.null(data.raw)) {
                    iter <- (1:dim(data.raw)[1]) - 1
                    data.raw.idx <- apply(data.raw, 1,
                        function(r) { !all(is.na(r)) })
                    data.raw <- data.raw[data.raw.idx,]
                    iter <- iter[data.raw.idx]
                    for (j in 1:dim(data.raw)[2]) {
                        val <- data.raw[,j]
                        val.eb <- if (length(iter) == 1) {
                                val
                            } else {
                                rep(NA, dim(data.raw)[1])
                            }
                        d.list[[length(d.list)+1]] <- data.frame(
                            iter=iter,
                            val=data.raw[,j],
                            val.eb=val.eb,
                            experiment=rep(experiment.name.legend, dim(data.raw)[1]),
                            dataset=rep(dataset.name, dim(data.raw)[1]),
                            run=rep(paste(dataset.name, stat.name, experiment.name, as.character(j), sep='/'), dim(data.raw)[1]),
                            stat=rep(stat.name.legend, dim(data.raw)[1]))
                    }
                }
            }
        }
    }

    d <- rbindlist(d.list)

    if (length(dim(d)) > 0 && dim(d)[1] > 0) {
        dir.create('plots')

        width <- 1.5*length(dataset.names) + 1
        height <- 1.5*length(stat.names) + 2

        for (j in 1:length(stat.names)) {
            filename.out <- paste('plots', paste(experiment.group.name, '_', stat.names[j], '.png', sep=''), sep='/')
            d.subset <- subset(d, stat == stat.names.legend[j])
            png(filename.out, width=5, height=5, pointsize=11, res=300, units='in')
            plot.tng3.smooth(d.subset)
            dev.off()
            #ggsave(filename.out, units='in', width=width, height=height)
        }

        #filename.out <- paste('plots', paste(experiment.group.name, '_raw.png', sep=''), sep='/')
        #ggplot(aes(x=iter, y=val, group=run, color=experiment, shape=experiment), data=d) +
        #    (if (length(stat.names) > 1 || length(dataset.names) > 1) {
        #            facet_grid(stat ~ dataset, scales='free')
        #        } else {
        #            NULL
        #        }) +
        #    geom_line(alpha=0.3, size=0.5) +
        #    theme_bw() +
        #    theme(legend.direction='horizontal', legend.position='bottom') +
        #    ylab('NMI') +
        #    xlab('document')
        #ggsave(filename.out, units='in', width=width, height=height)
    } else {
        cat('Data empty for', experiment.group.name, '\n')
    }
}

metrics <- c('out_of_sample_nmi')
metrics.legend <- c('out-of-sample NMI')

#plot.experiments('1', c('diff3', 'rel3', 'sim3'), c('1-rs0', '1-rs1k', '1-rs500k'), c('no rejuvenation', 'reservoir size 1k', 'full uniform rejuvenation'), metrics, metrics.legend)
plot.experiments('2', c('diff3', 'rel3', 'sim3'), c('2-rs1k-ibs0', '2-rs1k-ibs30', '2-rs1k-ibs100', '2-rs1k-ibs300', '2-rs1k-ibs3k'), c('no initialization', 'initialization size 30', 'initialization size 100', 'initialization size 300', 'batch gibbs'), metrics, metrics.legend)
#plot.experiments('2_3_4', c('diff3', 'rel3', 'sim3'), c('4', '3', '2-rs1k-ibs100'), c('no resample/rejuv', 'resample', 'resample and rejuv'), metrics, metrics.legend)
#plot.experiments('5', c('diff3', 'rel3', 'sim3'), c('5-ess5', '5-ess10', '5-ess20', '5-ess40'), c('ess 5', 'ess 10', 'ess 20', 'ess 40'), metrics, metrics.legend)
#plot.experiments('6_3', c('diff3', 'rel3', 'sim3'), c('3', '6-rs10k-rss10', '6-rs10k-rss30', '6-rs10k-rss100', '6-rs10k-rss300', '6-rs10k-rss1k'), c('no rejuv', 'rejuv size 10', 'rejuv size 30', 'rejuv size 100', 'rejuv size 300', 'rejuv size 1k'), metrics, metrics.legend)
#plot.experiments('8', c('diff3', 'rel3', 'sim3'), c('8-t2', '8-t3', '8-t4', '8-t5', '8-t6'), c('num topics 2', 'num topics 3', 'num topics 4', 'num topics 5', 'num topics 6'), metrics, metrics.legend)
#plot.experiments('9', c('diff3', 'rel3', 'sim3'), c('9-rs1k-ibs0', '9-rs1k-ibs10', '9-rs1k-ibs30', '9-rs1k-ibs100', '9-rs1k-ibs300', '9-rs1k-ibs1k', '9-rs1k-ibs3k'), c('no initialization', 'initialization size 10', 'initialization size 30', 'initialization size 100', 'initialization size 300', 'initialization size 1k', 'initialization size 3k'), metrics, metrics.legend)
#plot.experiments('10', c('diff3', 'rel3', 'sim3'), c('10-rs1k-ibs0', '10-rs1k-ibs10', '10-rs1k-ibs30', '10-rs1k-ibs100', '10-rs1k-ibs300', '10-rs1k-ibs1k', '10-rs1k-ibs3k'), c('no initialization', 'initialization size 10', 'initialization size 30', 'initialization size 100', 'initialization size 300', 'initialization size 1k', 'initialization size 3k'), metrics, metrics.legend)
plot.experiments('2_9_10', c('diff3', 'rel3', 'sim3'), c('2-rs1k-ibs100', '9-rs1k-ibs100', '10-rs1k-ibs100'), c('untuned init', 'nmi-tuned init', 'perplexity-tuned init'), metrics, metrics.legend)
#plot.experiments('11', c('null'), c('11-t3', '11-t6', '11-t12'), c('num topics 3', 'num topics 6', 'num topics 12'), c('out_of_sample_perplexity'), c('out-of-sample perplexity'))
#plot.experiments('12', c('null'), c('12-rs1k-ibs0', '12-rs1k-ibs10', '12-rs1k-ibs30', '12-rs1k-ibs100', '12-rs1k-ibs300', '12-rs1k-ibs1k', '12-rs1k-ibs3k'), c('no initialization', 'initialization size 10', 'initialization size 30', 'initialization size 100', 'initialization size 300', 'initialization size 1k', 'initialization size 3k'), c('out_of_sample_perplexity'), c('out-of-sample perplexity'))
#plot.experiments('13', c('null'), c('13-rs0', '13-rs10', '13-rs100', '13-rs1k', '13-rs10k'), c('no rejuvenation', 'reservoir size 10', 'reservoir size 100', 'reservoir size 1k', 'reservoir size 10k'), c('out_of_sample_perplexity'), c('out-of-sample perplexity'))
#plot.experiments('14', c('diff3', 'rel3', 'sim3'), c('14'), c('(ltr)'), metrics, metrics.legend)
#plot.experiments('15', c('diff3', 'rel3', 'sim3'), c('15-rs1k-a0.001', '15-rs1k-a0.01', '15-rs1k-a0.1', '15-rs1k-a1', '15-rs1k-a10'), c('alpha 0.001', 'alpha 0.01', 'alpha 0.1', 'alpha 1', 'alpha 10'), metrics, metrics.legend)
#plot.experiments('16', c('diff3', 'rel3', 'sim3'), c('16-rs1k-b0.001', '16-rs1k-b0.01', '16-rs1k-b0.1', '16-rs1k-b1', '16-rs1k-b10'), c('beta 0.001', 'beta 0.01', 'beta 0.1', 'beta 1', 'beta 10'), metrics, metrics.legend)
#plot.experiments('17', c('diff3', 'rel3', 'sim3'), c('17-rs0', '17-rs1k', '17-rs500k'), c('no rejuvenation', 'reservoir size 1k', 'full uniform rejuvenation'), metrics, metrics.legend)
#plot.experiments('18', c('diff3', 'rel3', 'sim3'), c('18-rs0', '18-rs1k', '18-rs500k'), c('no rejuvenation', 'reservoir size 1k', 'full uniform rejuvenation'), metrics, metrics.legend)
#plot.experiments('19', c('diff3', 'rel3', 'sim3'), c('19-rs0', '19-rs1k', '19-rs500k'), c('no rejuvenation', 'reservoir size 1k', 'full uniform rejuvenation'), metrics, metrics.legend)
plot.experiments('20', c('diff3', 'rel3', 'sim3'), c('20-rs0', '20-rs1k', '20-rs500k'), c('no rejuvenation', 'reservoir size 1k', 'full uniform rejuvenation'), metrics, metrics.legend)
#plot.experiments('21', c('diff3', 'rel3', 'sim3'), c('21-rs0', '21-rs1k', '21-rs500k'), c('no rejuvenation', 'reservoir size 1k', 'full uniform rejuvenation'), metrics, metrics.legend)
#plot.experiments('22', c('diff3', 'rel3', 'sim3'), c('22'), c('gibbs (csg init docs)'), metrics, metrics.legend)
#plot.experiments('23', c('diff3', 'rel3', 'sim3'), c('23'), c('gibbs (200 init docs)'), metrics, metrics.legend)
#plot.experiments('24', c('diff3', 'rel3', 'sim3'), c('24'), c('gibbs (100 init docs)'), metrics, metrics.legend)
#plot.experiments('25', c('diff3', 'rel3', 'sim3'), c('25-rs0'), c('no rejuvenation'), metrics, metrics.legend)
#plot.experiments('27-rs0', c('diff3', 'rel3', 'sim3'), c('27-rs0'), c('no rejuvenation'), metrics, metrics.legend)
#plot.experiments('27-rs1k', c('diff3', 'rel3', 'sim3'), c('27-rs1k'), c('reservoir size 1k'), metrics, metrics.legend)
#plot.experiments('28', c('diff3'), c('28-rs50000-rss5000', '28-rs100000-rss10000', '28-rs500000-rss50000'), c('reservoir size 50k', 'reservoir size 100k', 'reservoir size 500k'), metrics, metrics.legend)
#plot.experiments('29-a0.001', c('diff3'), c('29-a0.001-b0.001', '29-a0.001-b0.01', '29-a0.001-b0.1', '29-a0.001-b1', '29-a0.001-b10'), c('alpha 0.001, beta 0.001', 'alpha 0.001, beta 0.01', 'alpha 0.001, beta 0.1', 'alpha 0.001, beta 1', 'alpha 0.001, beta 10'), metrics, metrics.legend)
#plot.experiments('29-a0.01', c('diff3'), c('29-a0.01-b0.001', '29-a0.01-b0.01', '29-a0.01-b0.1', '29-a0.01-b1', '29-a0.01-b10'), c('alpha 0.01, beta 0.001', 'alpha 0.01, beta 0.01', 'alpha 0.01, beta 0.1', 'alpha 0.01, beta 1', 'alpha 0.01, beta 10'), metrics, metrics.legend)
#plot.experiments('29-a0.1', c('diff3'), c('29-a0.1-b0.001', '29-a0.1-b0.01', '29-a0.1-b0.1', '29-a0.1-b1', '29-a0.1-b10'), c('alpha 0.1, beta 0.001', 'alpha 0.1, beta 0.01', 'alpha 0.1, beta 0.1', 'alpha 0.1, beta 1', 'alpha 0.1, beta 10'), metrics, metrics.legend)
#plot.experiments('29-a1', c('diff3'), c('29-a1-b0.001', '29-a1-b0.01', '29-a1-b0.1', '29-a1-b1', '29-a1-b10'), c('alpha 1, beta 0.001', 'alpha 1, beta 0.01', 'alpha 1, beta 0.1', 'alpha 1, beta 1', 'alpha 1, beta 10'), metrics, metrics.legend)
#plot.experiments('29-a10', c('diff3'), c('29-a10-b0.001', '29-a10-b0.01', '29-a10-b0.1', '29-a10-b1', '29-a10-b10'), c('alpha 10, beta 0.001', 'alpha 10, beta 0.01', 'alpha 10, beta 0.1', 'alpha 10, beta 1', 'alpha 10, beta 10'), metrics, metrics.legend)
#plot.experiments('30', c('diff3'), c('30-rs1k', '30-rs10k', '30-rs100k', '30-rs500k'), c('reservoir size 1k', 'reservoir size 10k', 'reservoir size 100k', 'reservoir size 500k'), metrics, metrics.legend)
