args <- commandArgs(trailingOnly=TRUE)   # infile k outfile seed
suppressMessages(library(mclust))
d <- as.matrix(read.csv(args[1]))
set.seed(as.integer(args[4]))
m <- Mclust(d, G=as.integer(args[2]), verbose=FALSE)
write.table(m$classification, args[3], row.names=FALSE, col.names=FALSE)
