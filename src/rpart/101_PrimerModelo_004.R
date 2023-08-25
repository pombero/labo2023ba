# Arbol elemental con libreria  rpart
# Debe tener instaladas las librerias  data.table  ,  rpart  y  rpart.plot

# cargo las librerias que necesito
require("data.table")
require("rpart")
require("rpart.plot")
require("caret")
require("mlr")

# Aqui se debe poner la carpeta de la materia de SU computadora local
setwd("~/buckets/b1/") # Establezco el Working Directory

# cargo el dataset
dataset <- fread("./datasets/dataset_pequeno.csv")

dtrain <- dataset[foto_mes == 202107] # defino donde voy a entrenar
dapply <- dataset[foto_mes == 202109] # defino donde voy a aplicar el modelo

# Remove rows with missing values
#dtrain <- na.omit(dtrain)
#dapply <- na.omit(dapply)

# Define the parameter grid
#param_grid <- expand.grid(
#  cp = c(-0.3, -0.2, -0.1, 0.0, 0.001, 0.01, 0.1),
#  minsplit = c(0, 5, 10, 15, 20, 25, 30),
#  minbucket = c(1, 3, 5, 7, 9, 11, 13),
#  maxdepth = c(1, 3, 5, 7, 9, 11, 13)
#)

# Define the task
task <- makeClassifTask(data = as.data.frame(dtrain), target = "clase_ternaria")

# Define the learner
learner <- makeLearner("classif.rpart")

# Define the parameter set
param_set <- makeParamSet(
  makeNumericParam("cp", lower = -0.3, upper = 0.1),
  makeIntegerParam("minsplit", lower = 0, upper = 2),
  makeIntegerParam("minbucket", lower = 1, upper = 2),
  makeIntegerParam("maxdepth", lower = 1, upper = 5)
)

# Define the resampling strategy
rdesc <- makeResampleDesc("CV", iters = 10)

# Perform grid search using resampling
ctrl <- makeTuneControlGrid()
tune_result <- tuneParams(
  learner = learner,
  task = task,
  resampling = rdesc,
  measures = acc, # Use accuracy as the evaluation metric
  par.set = param_set,
  control = ctrl
)

# Get the best combination of parameters
best_params <- tune_result$x

# genero el modelo,  aqui se construye el arbol
# quiero predecir clase_ternaria a partir de el resto de las variables
modelo <- train(learner, task, subset = tune_result$opt.path$train.inds)
#modelo <- rpart(
#  clase_ternaria ~ .,
#  data = dtrain, # los datos donde voy a entrenar
#  control = rpart.control(
#    cp = best_params$cp,
    # esto significa no limitar la complejidad de los splits
#    minsplit = best_params$minsplit,
    # minima cantidad de registros para que se haga el split
#    minbucket = best_params$minbucket, # tamaño minimo de una hoja
#    maxdepth = best_params$maxdepth
#  )
#) # profundidad maxima del arbol

# Extract the underlying rpart model
#modelo_rpart <- getLearnerModel(modelo)$learner.model


# grafico el arbol
#prp(modelo_rpart,
#  extra = 101, digits = -5,
#  branch = 1, type = 4, varlen = 0, faclen = 0
#)


# aplico el modelo a los datos nuevos
prediccion <- predict(
  object = modelo,
  newdata = dapply,
  type = "prob"
)

# prediccion es una matriz con TRES columnas,
# llamadas "BAJA+1", "BAJA+2"  y "CONTINUA"
# cada columna es el vector de probabilidades

# agrego a dapply una columna nueva que es la probabilidad de BAJA+2
dapply[, prob_baja2 := prediccion[, "BAJA+2"]]

# solo le envio estimulo a los registros
#  con probabilidad de BAJA+2 mayor  a  1/40
dapply[, Predicted := as.numeric(prob_baja2 > 1 / 40)]

# genero el archivo para Kaggle
# primero creo la carpeta donde va el experimento
dir.create("./exp/")
dir.create("./exp/KA2001")

# solo los campos para Kaggle
fwrite(dapply[, list(numero_de_cliente, Predicted)],
  file = "./exp/KA2001/K101_004.csv",
  sep = ","
)
