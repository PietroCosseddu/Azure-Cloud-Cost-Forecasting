Nel corso dell'intero periodo dedicato a questo compito, ho testato un vasto numero di

modelli e tecniche al fine di comprendere e studiare quale fosse la soluzione ottimale.

Ho iniziato con modelli già noti e ampiamente testati, come XGBoost, per poi esplorare

nuove tecniche che integrano previsioni su singole categorie in modo intelligente.

Le due architetture principali testate sono state XGBoost, utilizzata come baseline

iniziale, e le reti neurali LSTM, particolarmente adatte per lavorare su serie

temporali.

In ogni prova, i dati forniti ai modelli consistevano nei costi dei giorni precedenti,

insieme ad altre informazioni temporali utili per aiutare il modello a predire, come il

giorno della settimana, il mese o il giorno dell'anno. Nel caso in cui avessi avuto a

disposizione un dataset più esteso, coprendo anni di costi, sarebbe stato vantaggioso

includere ulteriori informazioni come l'anno o i trimestri. Ho sperimentato diverse

configurazioni sul numero di giorni precedenti considerati dal modello, al fine di

individuare la configurazione ottimale per le diverse architetture testate.

Le principali metriche valutative considerate sono state due:

RMSE (Root Mean Square Error): una metrica standard per valutare le prestazioni

di una regressione.

Direzionalità dell'accuratezza (Directional accuracy): una metrica che misura se il

modello prevede correttamente un aumento dei costi, così come si manifesta nei

valori reali. Questa metrica è utile per penalizzare i modelli che semplicemente

traslano indietro di un giorno l'andamento dei valori reali.

**XGBoost Base:**

Inizialmente, i parametri utilizzati sono stati quelli standard, provenienti da un tutorial

per la creazione di notebook volti all'addestramento di modelli per il time series

forecasting. Successivamente, al fine di migliorare le prestazioni secondo le metriche

considerate, si è proceduto con il tuning degli iperparametri. Per il tuning, si è impiegata

la libreria HyperOpt, la quale mira a minimizzare una metrica specifica (in questo caso,

l'RMSE), esplorando numerose combinazioni di parametri per individuare quelli che

minimizzano tale metrica. Utilizzando tali parametri ottimizzati, è stato quindi addestrato

il modello.

Relazione di fine tirocinio

3



<a name="br4"></a> 

#PARAMETRI PROTAGONISTI DEL TUNING

parameters = {'colsample\_bytree': ,

'gamma': ,

'learning\_rate': ,

'max\_depth': ,

'min\_child\_weight': ,

'n\_estimators': ,

'subsample': }

Possiamo dire senza dubbio che tra le due prove quello che funziona quasi sempre

meglio è la seconda, ovvero il modello con i parametri tunati sia dal punto di vista

dell’RMSE che, ed è quello in cui si differenziano maggiormente, nella Directional

Accuracy.

**LSTM:**

Durante i test con le reti LSTM, sono state eseguite diverse prove iniziali variando le

architetture, tuttavia mantenendo un numero ridotto di layer. Tra le diverse

sperimentazioni sono state incluse l'utilizzo di livelli bidirezionali LSTM e l'impiego di

livelli convolutivi precedenti ai livelli LSTM per la trasmissione dei dati. Tuttavia, nelle

prove più recenti con i risultati più soddisfacenti, si è scelto di incrementare le

dimensioni delle reti LSTM, riducendo al contempo il numero di livelli dello stesso tipo

con un numero decrescente di unità (256, 128, 64, 64).

model = Sequential()

model.add(InputLayer((9, 1)))

model.add(LSTM(256, activation='relu', return\_sequences=True))

model.add(LSTM(128, activation='relu', return\_sequences=True))

model.add(LSTM(64, activation='relu', return\_sequences=True))

model.add(LSTM(64, activation='relu'))

model.add(Dense(1))

Per tutte le diverse prove condotte, l'architettura principale è stata sostanzialmente

simile: un maggior numero di livelli consente al modello di riconoscere più

efficacemente una vasta gamma di pattern nei dati durante l'addestramento. Se ci

fossero stati meno livelli, l'addestramento sarebbe stato più rapido ma molte

informazioni sarebbero potute andare perdute.

Relazione di fine tirocinio

4



<a name="br5"></a> 

Ogni approccio è stato testato considerando diverse quantità di giorni precedenti su cui

basare le previsioni (2-4-6). Inoltre, sono stati condotti 10 addestramenti e tra questi è

stato selezionato il modello sulla base della directional accuracy sul set di validazione.

L'approccio di addestramento è stato di tipo walk-forward, con 5 passaggi: ogni

passaggio comprendeva rispettivamente 50, 10 e 5 dati per i set di training, validation e

test. Le valutazioni delle metriche sono state ottenute dalla media delle metriche in

ciascuno dei 5 passaggi nell'addestramento.

Oltre ai giorni precedenti, i modelli hanno ricevuto informazioni temporali, come il mese

e il giorno della settimana da predire. Durante l'addestramento di ciascun approccio, è

stata utilizzata una metrica e una loss personalizzate che considerano sia l'RMSE che

la directional accuracy, permettendo un possibile aumento del peso di una delle due

metriche per conferirle maggiore importanza. È importante notare che durante gli

addestramenti, i pesi sono rimasti equamente distribuiti tra le due.

Per garantire un approccio il più robusto possibile, sono state effettuate 4 prove con le

architetture LSTM:

1\. Modello univariato basato sui costi totali giornalieri (somma di tutte le categorie).

2\. Modello multivariato basato sui costi delle singole categorie, effettuando previsioni

separate per ciascuna categoria e sommandole per ottenere il costo totale previsto.

3\. Modello multivariato simile al precedente, ma invece di sommare i costi, sono stati

utilizzati i risultati di un metamodello che unisce intelligentemente le previsioni di

tutte le categorie.

4\. Modello multivariato che considera i singoli valori di tutte le categorie insieme,

fornendo il costo totale senza una fase di somma separata, poiché un singolo

modello analizza tutti i valori delle diverse categorie.

Per quanto riguarda la divisione in categorie, il dataset conteneva costi relativi a più di

50 categorie, ma solo 8 di queste erano presenti nei dati per più del 90% dei giorni. Di

conseguenza, ho scelto di effettuare previsioni separate solo per le categorie presenti

quasi tutti i giorni e aggregare i costi delle categorie meno frequenti in una singola

previsione.

Tra i 4 approcci, il terzo è stato quello su cui ho concentrato maggiormente l'attenzione,

in quanto il problema principale risiedeva nella ricerca di un metamodello che

massimizzasse contemporaneamente le due metriche in questione (RMSE e DA). I

modelli testati sono stati:

Relazione di fine tirocinio

5



<a name="br6"></a> 

Stessa architettura LSTM utilizzata per l'addestramento e le singole previsioni.

Rete neurale con più livelli dense sovrapposti.

Modello XGBoost di base.

Modello XGBoost con tuning degli iperparametri.

Tra i 4 approcci, quello che ha ottenuto i risultati migliori è stato l'ultimo, in quanto il

tuning dei parametri è stato eseguito sui dati utilizzati per l'addestramento e mirato a

minimizzare sia l'RMSE che la DA.

**Parte 4 → Risultati e conclusioni:**

In conclusione ho creato delle tabelle contenenti tutti i valori delle due metriche dati da

tutte le prove che sono state fatte con i diversi modelli:

Models and approaches RMSE

Directional Accuracy

XGB base

XGB tuned

17\.450

14\.292

0\.26

0\.63

È importante notare che nella tabella, i valori forniti dal modello XGBoost con il tuning

degli iperparametri rappresentano i risultati ottenuti dall'addestramento migliore o dai

migliori parametri. Durante le varie prove, sono stati ottenuti anche risultati minori, ma si

è scelto di selezionare il miglior risultato tra tutte le prove. Questo principio si applica

anche a tutti gli approcci LSTM, in cui sono stati eseguiti diversi addestramenti. Nella

tabella, sono presentati i valori migliori sul test set, selezionati in base all'addestramento

con il risultato migliore ottenuto sul validation set.

Giorni indietro visibili ai

2 Days

4 Days

6 Days

modelli

RMSE = 11 DA =

0\.64

RMSE = 14,1 DA =

0\.48

RMSE = 17,3 DA =

0\.48

LSTM Approach 1

RMSE = 15,44 DA = RMSE = 17,2 DA =

RMSE = 14,63 DA =

0\.64

LSTM Approach 2

0\.68

0\.56

LSTM Approach 3 (XGB

TUNED, 2 days) (NN with

RMSE = 14,1 DA =

RMSE = 12,85 DA =

0\.68

RMSE = 15,30 DA =

0\.68

2 dense layer, 4 days) (NN 0.75

with 2 dense layer, 6 days

Relazione di fine tirocinio

6



<a name="br7"></a> 

RMSE = 12,9 DA =

0\.52

RMSE = 22 DA =

0\.48

RMSE = 18,28 DA =

0\.52

LSTM Approach 4

Inoltre è importante notare che, in ciascuno degli approcci, il risultato presentato è

quello ottenuto dall'addestramento migliore, che ha mostrato valori più elevati

soprattutto per quanto riguarda la directional accuracy. Riguardo al metamodello, in

particolare, i risultati, soprattutto nelle prove con 4 e 6 giorni, sono stati migliori, ma in

media non risultano altrettanto soddisfacenti, attestandosi intorno al 0.6 di accuratezza.

L'approccio che porta alle prestazioni migliori in termini di directional accuracy è quello

che utilizza il metamodello. Tuttavia, è importante notare che il metamodello XGBoost

con Hyperparameter Tuning sembra comportarsi meglio in più casi. Quando il tuning è

effettuato basandosi anche sul test set, i risultati sono migliori. Tuttavia, mostrando solo

i dati di addestramento e di validazione (durante la scelta dei parametri), i risultati

peggiorano all'aumentare del numero di step. Inoltre, l'approccio con i costi sommati

sembra funzionare in modo particolarmente efficace, ma non sembra prevedere

altrettanto bene un repentino aumento dei prezzi, aspetto che invece viene spesso

predetto in modo soddisfacente dal metamodello.

Relazione di fine tirocinio
