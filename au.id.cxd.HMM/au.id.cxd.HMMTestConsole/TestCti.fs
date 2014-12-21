

namespace au.id.cxd.HMMTestConsole

open au.id.cxd.HMM
open System
open System.IO
open au.id.cxd.HMM.HiddenMarkovModel
open System.Collections.Generic
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra


module TestCti =

    let test() =

        let testData = "/Users/cd/Google Drive/Math/Markov Model/au.id.cxd.HMM/au.id.cxd.HMMTestConsole/data/example_train_data.csv"
        let fInfo = new FileInfo(testData)
        let data = Reader.readSequences fInfo
        let states = Reader.readStates data
        let evidenceVars = Reader.readEvidenceVars data

        let test = Estimation.countStateFreq data states
        let pi = Estimation.statePriors data states
        // double check pi sums to 1
        let test1 = List.sum pi

        states
        let T = Estimation.countTransitions data states
        let A = Estimation.stateTransitions data states
        // should sum to 1.
        let test2 = A.RowSums().Sum() 

        // TODO: estimate sequence transition frequencies.
        let endSequences = Estimation.uniqueSequences data "Ended"


        let test3 = List.head endSequences
        // [string list list] list - a list of subsequences
        let subSeqs = Estimation.allSubsequences data endSequences

        let Bk = Estimation.avgPriorEvidences subSeqs evidenceVars states

        // so A and Bk and pi comprise the prior distributions for state transitions
        // A = P(x_t | x_t-1)
        // B = P(e_i | x_j)
        // pi = P(x_t)

        // and we have the labels for states and evidenceVars 

        let inputModel = {
            pi = pi;
            A = A;
            Bk = [Bk];
            states = states;
            evidence = evidenceVars;
            } 

        // let model = HiddenMarkovModel.train inputModel

        let testSeq = test3 |> List.toSeq |> Seq.take ((List.length test3) - 1) |> Seq.toList
        let testIdx = HiddenMarkovModel.indices testSeq evidenceVars
        let Barray = List.toArray [Bk]
        let stateCount = List.length states
        let evidenceCount = List.length evidenceVars

        let alpha1 = HiddenMarkovModel.alpha ((List.length states)-1) pi A Barray.[0] stateCount evidenceCount testIdx

        let testAlpha1 = alpha1.RowSums().Sum()

        let beta1 = HiddenMarkovModel.beta ((List.length states)-1) pi A Barray.[0] stateCount evidenceCount testIdx
        let testBeta1 = beta1.RowSums().Sum()

        // endSequences
        // test training on the end sequences
        // theta = 0.05
        // max epochs = 10

        let model = HiddenMarkovModel.train inputModel data 0.0005 50

        let testFile = "/Users/cd/Google Drive/Math/Markov Model/au.id.cxd.HMM/au.id.cxd.HMMTestConsole/data/testmodel.bin"
        let saveFlag = HiddenMarkovModel.writeToFile model (new FileInfo(testFile))
        let (readModel:Option<Model>) = HiddenMarkovModel.readFromFile (new FileInfo(testFile))

        // next steps prediction and decoding methods for HMM
        // note testSeq contains only evidence variables
        let prediction = 
            Option.map (fun model -> 
                        HiddenMarkovModel.predict model testSeq) 
                    readModel
        ()
