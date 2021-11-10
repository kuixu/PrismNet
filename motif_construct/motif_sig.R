##########################################################
#This R script is to analysis the motif enrichment and cluster
##########################################################

Args <- commandArgs()
in_file = Args[6]
out_file = Args[7]


human_motif<-read.table(file=in_file, header = F, sep = "\t")

head(human_motif)

i = 1
Pvalue <- c()
Odd_ratio <- c()
FDR <- c()
for(i in 1:dim(human_motif)[1]){
  Sum = human_motif[i,2]/human_motif[i,3]
  compare<-matrix(floor(c(human_motif[i,2],Sum*0.1,Sum*(1-human_motif[i,3]),Sum*0.9)),nr=2,dimnames=
                    list(c("sites","not sites"),c("motif","random")))
  fisher_test <- fisher.test(compare,alternative = "greater")
  Pvalue <- c(Pvalue, fisher_test$p.value)
  Odd_ratio <- c(Odd_ratio, fisher_test$estimate)
}

FDR <- p.adjust(Pvalue, method = "fdr")

colnames(human_motif) <- c("motif", "number", "percent")
data1 <- cbind(human_motif, Odd_ratio)
data1 <- cbind(data1, Pvalue)
data1 <- cbind(data1, FDR)
write.table(data1, file = out_file, row.names = F, col.names = T, sep = "\t")
