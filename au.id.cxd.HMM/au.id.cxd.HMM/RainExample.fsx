
#r "/Users/cd/Google Drive/Math/Markov Model/au.id.cxd.HMM/script/MathNet.Numerics.dll"
#r "/Users/cd/Google Drive/Math/Markov Model/au.id.cxd.HMM/script/MathNet.Numerics.FSharp.dll"
#load "Data/Reader.fs"
#load "Data/Estimation.fs"
#load "Model/HiddenMarkovModel.fs"

open au.id.cxd.HMM
open System
open System.IO
open au.id.cxd.HMM.HiddenMarkovModel
open System.Collections.Generic
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra

// includes a placeholder "final" evidence var and an "end" state

let data = 
    [["noumbrella"; "dry"];
     ["noumbrella"; "noumbrella"; "dry"];
     ["noumbrella"; "noumbrella"; "final"; "end"];
     ["noumbrella"; "noumbrella"; "noumbrella"; "dry"];
     ["noumbrella"; "noumbrella"; "noumbrella"; "final"; "end"];

     ["noumbrella"; "umbrella"; "rain"];
     ["noumbrella"; "umbrella"; "noumbrella"; "rain"];
     ["noumbrella"; "umbrella"; "noumbrella"; "noumbrella"; "rain"];
     ["noumbrella"; "umbrella"; "noumbrella"; "noumbrella"; "umbrella"; "rain"];
     ["noumbrella"; "umbrella"; "noumbrella"; "umbrella"; "rain"];
     ["noumbrella"; "umbrella"; "final"; "end"];

     ["umbrella"; "rain"];
     ["umbrella"; "umbrella"; "rain"];
     ["umbrella"; "umbrella"; "final"; "end"];
     ["umbrella"; "umbrella"; "umbrella"; "rain"];
     ["umbrella"; "umbrella"; "umbrella"; "final"; "end"];
     ]

let states = Reader.readStates data
let evidenceVars = Reader.readEvidenceVars data

let test = Estimation.countStateFreq data states
let pi = Estimation.statePriors data states

let T = Estimation.countTransitions data states
let A = Estimation.stateTransitions data states

// estimate sequence transition frequencies.
let endSequences = Estimation.uniqueSequences data "end"

let test3 = List.head endSequences
// [string list list] list - a list of subsequences
let subSeqs = Estimation.allSubsequences data endSequences

let Bk = Estimation.avgPriorEvidences subSeqs evidenceVars states


let inputModel = {
    pi = pi;
    A = A;
    Bk = [Bk];
    states = states;
    evidence = evidenceVars;
    } 

let endSequences3 = [["noumbrella"; "dry";];
                     ["umbrella"; "rain";];]

let stateCount = 3
let evidenceCount = 3

let alpha1 = HiddenMarkovModel.alpha ((List.length states)-1) pi A Bk stateCount evidenceCount [0;1;1;1;]

let beta1 = HiddenMarkovModel.beta (stateCount - 1) pi A Bk stateCount evidenceCount [0;1;1;1;]

//let alpha2 = HiddenMarkovModel.alpha ((List.length states)-1) pi A Bk.[0] stateCount evidenceCount [0;]

let model = HiddenMarkovModel.train inputModel data 0.0005 100

let predict = HiddenMarkovModel.predict model ["noumbrella"; "umbrella"; "umbrella"; "noumbrella"; "umbrella"; "umbrella";]

let predict2 = HiddenMarkovModel.predict model ["noumbrella"; "umbrella"; "noumbrella"; "umbrella"; "umbrella"; "umbrella"; "umbrella";]


let predict3 = HiddenMarkovModel.predict model ["umbrella"; ]


let predict4 = HiddenMarkovModel.predict model ["noumbrella"; ]
