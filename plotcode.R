#RStudio recommended

#install.packages("reshape2")
#install.packages("ggplot2")
#install.packages("magrittr")
#install.packages("dplyr")
#install.packages("here")


library(reshape2)
library(ggplot2)
library(magrittr)
library(dplyr)
library(here)

# Plot Figure 3

suweb <-c(0.061611374407582936, 0.46919431279620855, 0.16113744075829384, 0.16113744075829384, 0.14691943127962084)
bounded_rfxbms <- c(0.10408475458361952, 0.32130845510397155, 0.517032310841148, 0.030982782275912172, 0.019993851501324742, 0.0065978456940241495)

suweb <- append(suweb, 0)
names = c("EL-0", "EL-1", "EL-2", "EL-3", "EL-4", "Random")
data <- data.frame(rfxbms=bounded_rfxbms,names=names)
suwebdat <- data.frame(SUWEB=suweb)

tot = cbind(data,suwebdat)
tot = tot[,c(2,1,3)]

tot <- melt(tot)
#names(dfp1)[3] <- "percent"
ggplot(tot, aes(x = names, y= value, fill = variable), xlab="Age Group") +
  geom_bar(stat="identity", width=.5, position = "dodge") + 
  theme(legend.position = c(0.85,0.85), 
        axis.text.x = element_text(size=15),
        axis.text.y = element_text(size=15),
        axis.title.x = element_text(size=12),
        axis.title.y = element_text(size=12),
        legend.title = element_text(size=13),
        legend.text = element_text(size=14)
  ) +
  scale_y_continuous(limits = c(0, 0.6), breaks = c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)) +
  scale_fill_discrete(name="Fitting method",
                      breaks=c("rfxbms","SUWEB"),
                      labels=c("RFX-BMS","MLE on SUWEB")) + 
  xlab("Model") + ylab("Proportion of participants fitted")  +
  geom_text(data=tot[c(1,2,3,4,5,6),],aes(x=names,y=value,label=format(round(value, digits=3),nsmall = 3)),
            vjust=-0.32,size=4,hjust=1.1) + 
  geom_text(data=tot[c(7,8,9,10,11),],aes(x=names,y=value,label=format(round(value, digits=3),nsmall = 3)),
            vjust=-0.32,size=4,hjust=-0.1)

# Plot Figure 4

tom_rfxbms <- c(0.1628446565849932, 0.20955381145911933, 0.3689215648244876, 0.10117939982240508, 0.015289403238450365, 0.13697918692915367, 0.005231977141390788)
names = c("ToM-0","ToM-1","ToM-2","ToM-3","ToM-4","ToM-5","Random")
data <- data.frame(values=tom_rfxbms,names=names)
p <- ggplot(data=data, aes(x=names, y=values)) +
  geom_bar(stat="identity") +
  ylim(0,0.4) + 
  labs(x="",y="") + geom_text(data=data,aes(x=names,y=values,label=format(round(values, digits=3),nsmall = 3)),
                                                                    vjust=-1,size=5) +
  theme(axis.title.x=element_blank(),legend.title=element_blank(),axis.text=element_text(size=12))

p

# Plot Figure 5

setwd(substr(rstudioapi::getSourceEditorContext()$path,1,nchar(rstudioapi::getSourceEditorContext()$path)-11))
data <- read.csv2('correctrates_usecedegaoFalse.csv', header=FALSE)
tomall<-as.list(strsplit(data[8,], ",")[[1]])
tomall <- as.numeric(tomall)
tomall <- tomall[!is.na(tomall)]

tomrand<-as.list(strsplit(data[9,], ",")[[1]])
tomrand <- as.numeric(tomrand)
tomrand <- tomrand[!is.na(tomrand)]

data3 <- cbind(tomall)
data3 <- as.data.frame(unlist(data3[,1]))
rownames(data3) <- c() 
colnames(data3)<- c()

names(data3) <- "data3"

data3 <- data3 %>% 
  mutate(outlier = data3 > median(data3) + 
           IQR(data3)*1.5 | data3 < median(data3) -
           IQR(data3)*1.5) 
randdata <- data.frame(data3=tomrand,
                       outlier=TRUE)

data_norand <- data3[which(!(data3$data3 %in% tomrand)),]
data_norand <- data_norand[which(data_norand$outlier == TRUE),]

length(data3[which(data3$data3 > 0.736),][,1])
211 - length(data3[which(data3$data3 > 0.5),][,1])

data3 %>% 
  ggplot(aes("",data3),ylim=c(0,1),xlab="",xtitle="",ylab="",) + 
  geom_boxplot(outlier.shape = NA) + coord_cartesian(ylim = c(0, 1)) +
  geom_point(data = data_norand,#function(x) dplyr::filter(x, outlier), 
             position = "jitter") + geom_point(shape=4,data=randdata,position = "jitter") + 
  coord_flip() + scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) + 
  theme(
    axis.text.y=element_blank(),  #remove y axis labels
    axis.ticks.y=element_blank(),  #remove y axis ticks
    axis.text=element_text(size=12),
    axis.title.x=element_blank(),
    axis.title.y=element_blank()
  )

round(mean(data3[,1]),digits=3)
round(median(data3[,1]),digits=3)
round(IQR(data3[,1]),digits=3)
