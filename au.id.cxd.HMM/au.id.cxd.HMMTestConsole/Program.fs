namespace au.id.cxd.HMM

open System
open au.id.cxd.HMM
open System.IO
open au.id.cxd.HMM.HiddenMarkovModel
open au.id.cxd.HMMTestConsole

module Program = 

    

    [<EntryPoint>]
    let main args = 
        try 
            //TestRain2.test()
            TestRain.test()
            //TestCti.test()
        with 

        | e -> ()
        0
    

