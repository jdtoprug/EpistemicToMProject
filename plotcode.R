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
#bounded_rfxbms <- c(0.10408475458361952, 0.32130845510397155, 0.517032310841148, 0.030982782275912172, 0.019993851501324742, 0.0065978456940241495)  # Empty = nK
#bounded_rfxbms <- c(0.00545513224870438, 0.19190706245018682, 0.1551908216030796, 0.12417401520373965, 0.4417255941951964, 0.08154737429909313)  # Empty = k88
#bounded_rfxbms <- c(0.010849228473748298, 0.4297274005741532, 0.2009838843558826, 0.09020298316632629, 0.20535301539118328, 0.06288348803870636)  # Empty = K8A
#bounded_rfxbms <- c(0.005716309587470048, 0.2569947643741019, 0.09329331408991885, 0.10276072597432562, 0.4617843506434915, 0.07945053533069198)  # Empty = KAA
#bounded_rfxbms <- c(0.005141471068905926, 0.27015672899319504, 0.21601077747950412, 0.10971558855605856, 0.3265907435071386, 0.07238469039519771)  # Empty = 1/4 correct
#bounded_rfxbms <- c(0.005229311160828285, 0.293860958244113, 0.16900182936247457, 0.1124375610082579, 0.34340125626948587, 0.07606908395484036)  # Empty = randomize over K's
#bounded_rfxbms <- c(0.005234763250412799, 0.28741228124336, 0.27051654667438585, 0.09340795954682934, 0.2786055419868175, 0.06482290729819445)  # Empty = 1/2 over K/nK, 1/3 over each K
#bounded_rfxbms <- c(0.07687789083843914, 0.2918261165670651, 0.4305998839844988, 0.03029274635142112, 0.1641464820913098, 0.006256880167265942)  # Empty = nK, bugs fixed
#bounded_rfxbms <- c(0.00528147896828415, 0.12796027254339085, 0.1512377885743652, 0.07667518257303874, 0.5765952200597897, 0.06225005728113135) # Empty = k88, bug fixed

#bounded_rfxbms <- c(0.07687789083843914, 0.2918261165670651, 0.4305998839844988, 0.03029274635142112, 0.1641464820913098, 0.006256880167265942)  # Empty = nK, ALL fixed?
bounded_rfxbms <- c(0.009122738790207629, 0.35945181528224535, 0.21414424171969523, 0.07923524261043469, 0.2873222497446463, 0.05072371185277066)  # Empty = K8A, ALL FIXED?

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
  scale_y_continuous(limits = c(0, 0.6), breaks = c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)) +
  scale_fill_discrete(name="Fitting method",
                      breaks=c("rfxbms","SUWEB"),
                      labels=c("RFX-BMS","MLE on SUWEB")) + 
  xlab("Model") + ylab("Proportion of participants fitted")  +
  geom_text(data=tot[c(1,2,3,4,5,6),],aes(x=names,y=value,label=format(round(value, digits=3),nsmall = 3)),
            vjust=-0.32,size=5,hjust=1.1) + 
  geom_text(data=tot[c(7,8,9,10,11),],aes(x=names,y=value,label=format(round(value, digits=3),nsmall = 3)),
            vjust=-0.32,size=5,hjust=-0.1)

# Plot Figure 4

#RFX-BMS results for ToM models, sorted as ToM-0, -1, -2, -3, -4, -5, random model
#tom_rfxbms <- c(0.1628446565849932, 0.20955381145911933, 0.3689215648244876, 0.10117939982240508, 0.015289403238450365, 0.13697918692915367, 0.005231977141390788)  # Empty = nK
#tom_rfxbms <- c(0.2597529671653401, 0.07999909658261765, 0.1710419956439245, 0.06922491941977701, 0.027721116397650125, 0.3870781501050217, 0.005181754685668923)  # Empty = K88
#tom_rfxbms <- c(0.19169495377172316, 0.30931325306083357, 0.2080373347295559, 0.02442992276432699, 0.01011109515714701, 0.2514161057512701, 0.004997334765143318)  # Empty = K8A
#tom_rfxbms <- c(0.2543002198693454, 0.10337278275350086, 0.1617154535485054, 0.05692963632511191, 0.02765205485933389, 0.3908460211095996, 0.0051838315346030795)  # Empty = KAA
#tom_rfxbms <- c(0.24593214594043755, 0.08750306044806072, 0.23585390830386757, 0.053747591441489675, 0.02104608618830192, 0.35074715657693883, 0.005170051100903686)  # Empty = 1/4 correct
#tom_rfxbms <- c(0.24191513198715317, 0.12597865697405583, 0.20708945609777138, 0.04304025931308752, 0.021565192348987183, 0.3552604667984596, 0.005150836480485396)  # Empty = randomize over K answers
#tom_rfxbms <- c(0.24266312631001866, 0.07667154696841322, 0.2639812709103868, 0.06758798013703103, 0.019485580016250996, 0.3244265724789119, 0.0051839231789874395)  # Empty = 1/2 over K/nK, 1/3 over each K
#tom_rfxbms <- c(0.1628446565849932, 0.20955381145911933, 0.3689215648244876, 0.10117939982240508, 0.015289403238450365, 0.13697918692915367, 0.005231977141390788)  # Empty = nK, bug fixed
#tom_rfxbms <- c(0.2597529671653401, 0.07999909658261765, 0.1710419956439245, 0.06922491941977701, 0.027721116397650125, 0.3870781501050217, 0.005181754685668923)  # Empty = K88, bug fixed

#tom_rfxbms <- c(0.1628446565849932, 0.20955381145911933, 0.3689215648244876, 0.10117939982240508, 0.015289403238450365, 0.13697918692915367, 0.005231977141390788)  # Empty = nK, ALL FIXED?
tom_rfxbms <- c(0.19169495377172316, 0.30931325306083357, 0.2080373347295559, 0.02442992276432699, 0.01011109515714701, 0.2514161057512701, 0.004997334765143318)  # Empty = K8A, ALL FIXED?

names = c("ToM-0","ToM-1","ToM-2","ToM-3","ToM-4","ToM-5","Random") #Model names
data <- data.frame(values=tom_rfxbms,names=names) #Convert to data frame

#Plot Figure 4
p <- ggplot(data=data, aes(x=names, y=values)) +
  geom_bar(stat="identity") +
  ylim(0,0.4) + 
  labs(x="",y="") + geom_text(data=data,aes(x=names,y=values,label=format(round(values, digits=3),nsmall = 3)),
                                                                    vjust=-1,size=5) +
  theme(axis.title.x=element_blank(),legend.title=element_blank(),axis.text=element_text(size=12))

#Show Figure 4
p

# Plot Figure 5

#Set working directory to plotcode.R file location (requires RStudio - without RStudio please set working directory to location of correctrates_usecedegaoFalse.csv manually)
setwd(substr(rstudioapi::getSourceEditorContext()$path,1,nchar(rstudioapi::getSourceEditorContext()$path)-11))
data <- read.csv2('correctrates_usecedegaoFalse.csv', header=FALSE)  #Read coherence data, change to 'usecedegaoTrue' to investigate Cedegao's coherence

#Read coherence for all participants, convert to numeric, and remove NA values
tomall<-as.list(strsplit(data[8,], ",")[[1]])
tomall <- as.numeric(tomall)
tomall <- tomall[!is.na(tomall)]

#Read coherence for participants where random model fits best, convert to numeric, and remove NA values
tomrand<-as.list(strsplit(data[9,], ",")[[1]])
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
