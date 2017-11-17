(ns cnet.core
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]))

;;Neurons
;; Input  Hidden  Output
;; A      1       C
;; B      2       D
;;        3

;;Connection strengths
;; Input to Hidden => [[A1 A2 A3][B1 B2 B3]]
;; Hidden to Output => [[1C 1D][2C 2D][3C 3D]]

  (def input-neurons [1 0])

  (def input-hidden-strengths [[0.12 0.2 0.13]
                               [0.01 0.02 0.03]])

  (def hidden-neurons [0 0 0])

  (def hidden-output-strengths [[0.15 0.16]
                                [0.02 0.03]
                                [0.01 0.02]])

  (def activation-fn (fn[x] (Math/tanh x)))

  (def dactivation-fn (fn[y] (- 1.0 (* y y))))

  (defn layer-activation [inputs strengths]
    "forward propogate the input of a layer"
    (mapv activation-fn
          (mapv #(reduce + %)
                (* inputs (transpose strengths)))))

(def new-hidden-neurons
  "hidden neuron values"
  (layer-activation input-neurons input-hidden-strengths))

(def new-output-neurons
  "output neuron values"
  (layer-activation new-hidden-neurons hidden-output-strengths))

(defn -main
  "Implememnting a neural network" ;;http://gigasquidsoftware.com/blog/2013/12/02/neural-networks-in-clojure-with-core-dot-matrix/
  [& args]
  (println "Implementing a neural network based on the tutorial at: http://gigasquidsoftware.com/blog/2013/12/02/neural-networks-in-clojure-with-core-dot-matrix/")


  (println (layer-activation input-neurons input-hidden-strengths))
  (println (layer-activation new-hidden-neurons hidden-output-strengths))


  )
