library(ggplot2)

options(warn=1)

stat.names <- c('in_sample_nmi', 'out_of_sample_nmi', 'out_of_sample_log_likelihood', 'out_of_sample_perplexity')

plot.experiments <- function(experiment.group.name, dataset.names, experiment.names, experiment.names.legend) {
    if (is.null(experiment.names.legend)) {
        experiment.names.legend <- experiment.names
    }

    cat(experiment.group.name, '\n')

    for (experiment.name in experiment.names) {
        extract.cmd <- paste('python', 'extract_log_stats.py', experiment.name, sep=' ')
        cat(extract.cmd, '\n')
        system(extract.cmd)
    }

    for (dataset.name in dataset.names) {
        cat('*', dataset.name, '\n')
        for (stat.name in stat.names) {
            data <- data.frame()
            for (i in 1:length(experiment.names)) {
                experiment.name <- experiment.names[i]
                experiment.name.legend <- experiment.names.legend[i]
                cat('  -', experiment.name, stat.name, '\n')
                filename.in <- paste(experiment.name, '/', dataset.name, '_', stat.name, '.tab', sep='')
                my.data.raw <- tryCatch(read.table(filename.in, header=T), error=function(ex) {NULL})
                if (! is.null(my.data.raw)) {
                    my.data <- data.frame(mean=rowSums(my.data.raw, na.rm=T)/rowSums(!is.na(my.data.raw)))
                    my.data$experiment <- rep(experiment.name.legend, dim(my.data)[1])
                    my.data$sd <- apply(my.data.raw, 1, function(r) { sd(r, na.rm=T) })
                    my.data$idx <- (1:dim(my.data)[1]) - 1
                    my.data$lcl <- my.data$mean - my.data$sd
                    my.data$ucl <- my.data$mean + my.data$sd
                    my.data <- my.data[apply(my.data.raw, 1, function(r) { !all(is.na(r)) }),]
                    data <- rbind(data, my.data)
                }
            }

            dir.create('plots')
            dir.create(paste('plots', experiment.group.name, sep='/'))
            filename.out <- paste('plots', experiment.group.name, paste(dataset.name, '_', stat.name, '.png', sep=''), sep='/')
            if (dim(data)[1] > 0) {
                qplot(idx, mean, data=data, group=experiment) + geom_smooth(aes(fill=experiment, ymin=lcl, ymax=ucl, color=experiment), data=data, stat="identity") + ylab(paste(stat.name, '(mean +/- stdev)')) + xlab('document') + ggtitle(paste(dataset.name, stat.name)) #+ ylim(0,1)
                ggsave(filename.out)
            } else {
                cat('Data empty for', filename.out, '\n')
            }
        }
    }
}

plot.experiments('1', c('diff3', 'rel3', 'sim3'), c('1-rs0', '1-rs100', '1-rs1k', '1-rs10k', '1-rs100k', '1-rs500k'), c('reservoir size 0', 'reservoir size 100', 'reservoir size 1k', 'reservoir size 10k', 'reservoir size 100k', 'reservoir size 500k'))
plot.experiments('2', c('diff3', 'rel3', 'sim3'), c('2-rs1k-ibs0', '2-rs1k-ibs10', '2-rs1k-ibs30', '2-rs1k-ibs100', '2-rs1k-ibs300', '2-rs1k-ibs1k', '2-rs1k-ibs3k'), c('no initialization', 'initialization size 10', 'initialization size 30', 'initialization size 100', 'initialization size 300', 'initialization size 1k', 'initialization size 3k'))
plot.experiments('2_3_4_7', c('diff3', 'rel3', 'sim3'), c('4', '7', '3', '2-rs1k-ibs100'), c('no resample/rejuv', 'resample-propagate', 'resample', 'resample and rejuv'))
plot.experiments('5', c('diff3', 'rel3', 'sim3'), c('5-ess5', '5-ess10', '5-ess20', '5-ess40'), c('ess 5', 'ess 10', 'ess 20', 'ess 40'))
plot.experiments('6', c('diff3', 'rel3', 'sim3'), c('6-rs10k-rss10', '6-rs10k-rss30', '6-rs10k-rss100', '6-rs10k-rss300', '6-rs10k-rs1k'), c('rejuv size 10', 'rejuv size 30', 'rejuv size 100', 'rejuv size 300', 'rejuv size 1k'))
plot.experiments('8', c('diff3', 'rel3', 'sim3'), c('8-t2', '8-t3', '8-t4', '8-t5', '8-t6'), c('num topics 2', 'num topics 3', 'num topics 4', 'num topics 5', 'num topics 6'))
plot.experiments('9', c('diff3', 'rel3', 'sim3'), c('9-rs1k-ibs0', '9-rs1k-ibs10', '9-rs1k-ibs30', '9-rs1k-ibs100', '9-rs1k-ibs300', '9-rs1k-ibs1k', '9-rs1k-ibs3k'), c('no initialization', 'initialization size 10', 'initialization size 30', 'initialization size 100', 'initialization size 300', 'initialization size 1k', 'initialization size 1k'))
plot.experiments('10', c('diff3', 'rel3', 'sim3'), c('10-rs1k-ibs0', '10-rs1k-ibs10', '10-rs1k-ibs30', '10-rs1k-ibs100', '10-rs1k-ibs300', '10-rs1k-ibs1k', '10-rs1k-ibs3k'), c('initialization size 0', 'initialization size 10', 'initialization size 30', 'initialization size 100', 'initialization size 300', 'initialization size 1k', 'initialization size 1k'))
plot.experiments('11', c('null'), c('11-t3', '11-t6', '11-t12'), NULL)
plot.experiments('12', c('null'), c('12-rs1k-ibs0', '12-rs1k-ibs10', '12-rs1k-ibs30', '12-rs1k-ibs100', '12-rs1k-ibs300', '12-rs1k-ibs1k', '12-rs1k-ibs3k'), NULL)
plot.experiments('13', c('null'), c('13-rs0', '13-rs10', '13-rs100', '13-rs1k', '13-rs10k'), NULL)
plot.experiments('14', c('diff3', 'rel3', 'sim3'), c('14'), NULL)
plot.experiments('15', c('diff3', 'rel3', 'sim3'), c('15-rs1k-a0.001', '15-rs1k-a0.01', '15-rs1k-a0.1', '15-rs1k-a1', '15-rs1k-a10'), NULL)
plot.experiments('16', c('diff3', 'rel3', 'sim3'), c('16-rs1k-b0.001', '16-rs1k-b0.01', '16-rs1k-b0.1', '16-rs1k-b1', '16-rs1k-b10'), NULL)
