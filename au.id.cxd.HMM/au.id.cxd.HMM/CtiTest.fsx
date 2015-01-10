// Learn more about F# at http://fsharp.net. See the 'F# Tutorial' project
// for more guidance on F# programming.

#r "/Users/cd/Google Drive/Math/Markov Model/au.id.cxd.HMM/script/MathNet.Numerics.dll"
#r "/Users/cd/Google Drive/Math/Markov Model/au.id.cxd.HMM/script/MathNet.Numerics.FSharp.dll"
#load "Data/Reader.fs"
#load "Data/Estimation.fs"
#load "Model/DataTypes.fs"
#load "Model/HiddenMarkovModel.fs"



open au.id.cxd.HMM
open System
open System.IO
open au.id.cxd.HMM.HiddenMarkovModel
open System.Collections.Generic
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open au.id.cxd.HMM.DataTypes


let testData = "/Users/cd/Google Drive/Math/Markov Model/au.id.cxd.HMM/au.id.cxd.HMMTestConsole/data/example_train_data.csv"

let fInfo = new FileInfo(testData)
let data = Reader.readSequences fInfo
let states = Reader.readStates data
let evidenceVars = Reader.readEvidenceVars data

let pi = Estimation.statePriors data states
let A = Estimation.stateTransitions data states
// TODO: estimate sequence transition frequencies.

let Bk = [Estimation.avgPriorEvidences data evidenceVars states]
//let Bk = [Estimation.jointPriorEvidences subSeqs evidenceVars states]

//let Bk = Estimation.priorEvidences subSeqs evidenceVars states

// so A and Bk and pi comprise the prior distributions for state transitions
// A = P(x_t | x_t-1)
// B = P(e_i | x_j)
// pi = P(x_t)

// and we have the labels for states and evidenceVars 

let inputModel = {
    pi = pi;
    A = A;
    Bk = Bk;
    states = states;
    evidence = evidenceVars;
    } 

// let model = HiddenMarkovModel.train inputModel

let stateCount = List.length states
let evidenceCount = List.length evidenceVars

// endSequences
// test training on the end sequences
// theta = 0.05
// max epochs = 10

let model = HiddenMarkovModel.train inputModel data 0.00001 10

let test5 = ["Ringing(inbound)";"UserEvent(Start)";"UserEvent(Stop)";"OffHook";"Established";"Held";"Dialing(Consult)"];
HiddenMarkovModel.viterbiPredict model test5

let test6 = ["Ringing(inbound)";"UserEvent(Start)";"UserEvent(Stop)";"OffHook";"Established";"Held";];
HiddenMarkovModel.viterbiPredict model test6

let test7 = ["Ringing(inbound)";"UserEvent(Start)";"UserEvent(Stop)";"OffHook";"Established";"Held";"Released";];
HiddenMarkovModel.viterbiPredict model test7
     

