

namespace au.id.cxd.HMM
open System
open System
open System.IO
open System.Text
open System.Collections.Generic
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open System.Runtime.Serialization
open System.Runtime.Serialization.Formatters.Binary
open au.id.cxd.HMM.DataTypes

module Serialiser =

    (* 
    write the supplied model to file.
    *)
    let writeToFile (model:'a) (file:FileInfo) : bool =
        let binFormat = new BinaryFormatter()
        try 
            let outFile = new FileStream(file.FullName, FileMode.Create, FileAccess.Write, FileShare.None)
            binFormat.Serialize(outFile, model)
            outFile.Close()
            true
        with _ -> false

    (*
    read the supplied model from file
    *)
    let readFromFile (file:FileInfo) : Option<'a> =
        let binFormat = new BinaryFormatter()
        try
            let inFile = new FileStream(file.FullName, FileMode.Open, FileAccess.Read, FileShare.Read)
            let data = binFormat.Deserialize(inFile) :?> 'a
            inFile.Close()
            Some(data)
        with _ -> None

