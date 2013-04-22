
import Obsidian

import Data.Word
import Data.Int
import Data.Bits

import Prelude hiding (zipWith,sum,replicate,take,drop)
import qualified Prelude as P 
---------------------------------------------------------------------------
-- Util 
---------------------------------------------------------------------------
quickPrint :: ToProgram a b => (a -> b) -> Ips a b -> IO ()
quickPrint prg input =
  putStrLn $ genKernel "kernel" prg input
  
---------------------------------------------------------------------------
-- Scans 
---------------------------------------------------------------------------
sklansky :: (Choice a, MemoryOps a)
            => Int
            -> (a -> a -> a)
            -> Pull Word32 a
            -> BProgram (Pull Word32 a)
sklansky 0 op arr = return arr
sklansky n op arr =
  do 
    let arr1 = binSplit (n-1) (fan op) arr
    arr2 <- force arr1
    sklansky (n-1) op arr2

-- fan :: (Choice a, ASize l) => (a -> a -> a) -> Pull l a -> Pull l a
fan :: Choice a => (a -> a -> a) -> SPull a -> SPull a
fan op arr =  a1 `conc`  fmap (op c) a2 
    where 
      (a1,a2) = halve arr
      c = a1 ! sizeConv (len a1 - 1)

splitUp :: (ASize l, Num l)
           => l -> Pull (EWord32) a -> Pull (EWord32) (Pull l a)
splitUp n (Pull m ixf) = Pull (m `div` fromIntegral n) $ 
                          \i -> Pull n $ \j -> ixf (i * (sizeConv n) + j)


sklanskyG logbs op arr =
  mapG (sklansky logbs op) (splitUp (2^logbs) arr)

getSklansky =
  quickPrint (sklanskyG 5 (+))
             (undefinedGlobal (variable "X") :: DPull EFloat)