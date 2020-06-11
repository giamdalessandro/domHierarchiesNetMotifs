# DomWorld import script. Do not edit
# Source me in the directory of the *.csv files!

Runs <- list()
for (r in 1:3) 
{
    Runs <- rbind(Runs, read.csv(paste0("output", r, ".csv"), sep="\t", header=TRUE))
}
write.table(Runs, file="FILENAME.csv", sep = ";", na = "NA", dec = ".", row.names = FALSE)
