#RStudio recommended

#install.packages("reshape2")
#install.packages("ggplot2")
#install.packages("magrittr")
#install.packages("dplyr")

#Load require libraries
library(reshape2)
library(ggplot2)
library(magrittr)
library(dplyr)

# Plot Figure 3

#For EL-0, -1, -2, -3, -4, respectively, proportion of Cedegao's participants that got fitted to that level for SUWEB
suweb <-c(0.061611374407582936, 0.46919431279620855, 0.16113744075829384, 0.16113744075829384, 0.14691943127962084)

#For, respectively, EL-0, -1, -2, -3, -4, and the random model, predicted frequency of that model in the population when non-stochastic SUWEB is fitted to Cedegao et al (2021)'s participants using RFX-BMS
bounded_rfxbms <- c(0.07687789083843914, 0.2918261165670651, 0.4305998839844988, 0.03029274635142112, 0.1641464820913098, 0.006256880167265942)  # Empty = nK
bounded_rfxbms <- c(0.005883180220330229, 0.14536532972030333, 0.2943418493072698, 0.04403716830785662, 0.13546336448388752, 0.37490910796035254)  # Error 0.25, REMOVE
bounded_rfxbms <- c(0.10504828638864677, 0.2732779208202732, 0.3992071997454847, 0.051486844137903706, 0.16616646387557676, 0.004813285032114883)  # 1/4, new random l, REMOVE
bounded_rfxbms <- c(0.07800296276913084, 0.2922452118537038, 0.43036451956613075, 0.030247845783120723, 0.16421448138803446, 0.004924978639879472)  # penalty 0.5, new random l, REMOVE
bounded_rfxbms <- c(0.016789217238976, 0.22437157639600183, 0.3839997612673433, 0.04311531829289777, 0.15228764285556892, 0.17943648394921205)  # 1/3, old random l, REMOVE
bounded_rfxbms <- c(0.09430506263986635, 0.28091282192258143, 0.41112906490732204, 0.04213854448698609, 0.16669655229032299, 0.0048179537529212104)  # 1/3, new random l, REMOVE

suweb <- append(suweb, 0) #SUWEB has no random model so proportion of random model for SUWEB is 0
names = c("EL-0", "EL-1", "EL-2", "EL-3", "EL-4", "Random") #Names of the models
data <- data.frame(rfxbms=bounded_rfxbms,names=names)  #Convert bounded_rfxbms to data frame
suwebdat <- data.frame(SUWEB=suweb) #Convert suweb to dta frame

tot = cbind(data,suwebdat) #Combine both fits
tot = tot[,c(2,1,3)] #Re-order columns

tot <- melt(tot)  # Combine columns with proportion

# Plot Figure 3
ggplot(tot, aes(x = names, y= value, fill = variable), xlab="Age Group") +
  geom_bar(stat="identity", width=.5, position = "dodge") + 
  theme(legend.position = c(0.85,0.85), 
        axis.text.x = element_text(size=15),
        axis.text.y = element_text(size=15),
        axis.title.x = element_text(size=15),
        axis.title.y = element_text(size=15),
        legend.title = element_text(size=13),
        legend.text = element_text(size=14)
  ) +
  scale_y_continuous(limits = c(0, 0.55), breaks = c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)) +
  scale_fill_discrete(name="Fitting method",
                      breaks=c("rfxbms","SUWEB"),
                      labels=c("RFX-BMS","MLE on SUWEB")) + 
  xlab("Model") + ylab("Proportion of participants fitted")  +
  geom_text(data=tot[c(1,2,3,4,5,6),],aes(x=names,y=value,label=format(round(value, digits=3),nsmall = 3)),
            vjust=-0.32,size=5,hjust=1.1) + 
  geom_text(data=tot[c(7,8,9,11),],aes(x=names,y=value,label=format(round(value, digits=3),nsmall = 3)),
            vjust=-0.32,size=5,hjust=-0.1) +
  geom_text(data=tot[c(10),],aes(x=names,y=value,label=format(round(value, digits=3),nsmall = 3)),
            vjust=-0.32,size=5,hjust=0.5)
# Plot Figure 4

#RFX-BMS results for ToM models, sorted as ToM-0, -1, -2, -3, -4, -5, random model
tom_rfxbms <- c(0.1628446565849932, 0.20955381145911933, 0.3689215648244876, 0.10117939982240508, 0.015289403238450365, 0.13697918692915367, 0.005231977141390788)  # Empty = nK
tom_rfxbms <- c(0.012383939634639908, 0.1409398219824671, 0.2754670935232291, 0.08690114633458751, 0.016413434646214885, 0.1131562065072192, 0.35473835737164217)  # Error 0.25, REMOVE
tom_rfxbms <- c(0.19450948460794854, 0.19347474465505868, 0.3410626131695885, 0.1135435579562663, 0.017607526301440773, 0.1350203989237588, 0.004781674385938458)  # 1/4, new random l, REMOVE
tom_rfxbms <- c(0.16318806878908, 0.2095751006072679, 0.36884512332818414, 0.10120503483293579, 0.015292863944595343, 0.1370300319651128, 0.004863776532823963)  # Penalty 0.5, new random l, REMOVE
tom_rfxbms <- c(0.12922471630186533, 0.18940954906164148, 0.3413314313092467, 0.10408314779891689, 0.015897936190540397, 0.12802734001918925, 0.09202587931859998)  # 1/3 penalty, old random l, REMOVE
tom_rfxbms <- c(0.18404549585774072, 0.19831837972901023, 0.3515981104446551, 0.10938312606588221, 0.016394877256068027, 0.13548510516638113, 0.0047749054802626945)  # 1/3 penalty, new random algo, REMOVE

names = c("ToM-0","ToM-1","ToM-2","ToM-3","ToM-4","ToM-5","Random") #Model names
names <- factor(names, levels = names) #Turn into ordered factor so ggplot doesn't sort the bars
data <- data.frame(values=tom_rfxbms,names=names) #Convert to data frame

#Plot Figure 4
p <- ggplot(data=data, aes(x=names, y=values)) +
  geom_bar(stat="identity") +
  scale_y_continuous(limits = c(0, 0.55), breaks = c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)) + 
  labs(x="",y="") + geom_text(data=data,aes(x=names,y=values,label=format(round(values, digits=3),nsmall = 3)),
                                                                    vjust=-1,size=5) +
  xlab("Model") + ylab("Proportion of participants fitted")  +
  theme(legend.title=element_blank(), 
      axis.text.x = element_text(size=15),
      axis.text.y = element_text(size=15),
      axis.title.x = element_text(size=15),
      axis.title.y = element_text(size=15),
      legend.text = element_text(size=14))

#Show Figure 4
p

# Plot Figure 5

#Set working directory to plotcode.R file location (requires RStudio - without RStudio please set working directory to location of correctrates_usecedegaoFalse.csv manually)
setwd(substr(rstudioapi::getSourceEditorContext()$path,1,nchar(rstudioapi::getSourceEditorContext()$path)-11))
data <- read.csv2('correctrates_usecedegaoFalse.csv', header=FALSE)  #Read coherence data, change to 'usecedegaoTrue' to investigate Cedegao's coherence

#Read coherence for all participants, convert to numeric, and remove NA values
tomall<-as.list(strsplit(data[nrow(data)-1,], ",")[[1]])
tomall <- as.numeric(tomall)
tomall <- tomall[!is.na(tomall)]

#Read coherence for participants where random model fits best, convert to numeric, and remove NA values
tomrand<-as.list(strsplit(data[nrow(data),], ",")[[1]])
tomrand <- as.numeric(tomrand)
tomrand <- tomrand[!is.na(tomrand)]

data3 <- cbind(tomall) # Convert to column
data3 <- as.data.frame(unlist(data3[,1])) #Convert to data frame
rownames(data3) <- c() # Remove row names
colnames(data3)<- c() # Remove column names

names(data3) <- "data3" # Name only column `data3' so we can refer to it

# Mark outliers
data3 <- data3 %>% 
  mutate(outlier = data3 > median(data3) + 
           IQR(data3)*1.5 | data3 < median(data3) -
           IQR(data3)*1.5) 

# List of outliers where random model fits best
randdata <- data.frame(data3=tomrand,
                       outlier=TRUE)

data_norand <- data3[which(!(data3$data3 %in% tomrand)),]  # Data without participants where random model fits best
data_norand <- data_norand[which(data_norand$outlier == TRUE),]  # Outliers where ToM model fits best

length(data3[which(data3$data3 > 0.736),][,1])  # How many participants have a coherence > 0.736 ?
211 - length(data3[which(data3$data3 > 0.5),][,1])  # How many participants have a coherence > 0.5 ?

# Create Figure 5
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

# Calculate mean, median, and IQR for coherence
round(mean(data3[,1]),digits=3)
round(median(data3[,1]),digits=3)
round(IQR(data3[,1]),digits=3)
