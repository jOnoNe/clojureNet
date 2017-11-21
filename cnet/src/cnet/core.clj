(ns cnet.core
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]))

;;Neurons
;; Input  Hidden  Output
;; A      1       C
;; B      2       D
;;        3

;; 2 input neurons
  (def input-neurons [1 0])

;; Input to Hidden => [[A1 A2 A3][B1 B2 B3]]
  (def input-hidden-strengths [[0.12 0.2 0.13]
                               [0.01 0.02 0.03]])

;; 3 middle neurons
  (def hidden-neurons [0 0 0])

;; Hidden to Output => [[1C 1D][2C 2D][3C 3D]]
  (def hidden-output-strengths [[0.15 0.16]
                                [0.02 0.03]
                                [0.01 0.02]])

;; Feed forward
  (def activation-fn (fn[x] (Math/tanh x))) ;; will output in the range between 1 and -1

;; for calculating errors
  (def dactivation-fn (fn[y] (- 1.0 (* y y))))

;; (Sum of all the inputs to a neuron) x (connection strength)
  (defn layer-activation [inputs strengths]
    "forward propogate the input of a layer"
    (mapv activation-fn ;;finally, fit to a range of 1 to -1
          (mapv #(reduce + %);; sum these together
                (* inputs (transpose strengths)))));; multiply inputs and strengths

(def new-hidden-neurons ;;work out the new hidden neuron strengths
  "hidden neuron values"
  (layer-activation input-neurons input-hidden-strengths))

(def new-output-neurons ;; work out the new output neuron strengths
  "output neuron values"
  (layer-activation new-hidden-neurons hidden-output-strengths))


(def targets [0 1])

(defn output-deltas [target outputs]
  "measures the delta errors for the output layer (Desired value - actual value) and multiplying it by the gradient of the activation function"
  (* (mapv dactivation-fn outputs)
     (- targets outputs)))


(defn -main
  "Implememnting a neural network" ;;http://gigasquidsoftware.com/blog/2013/12/02/neural-networks-in-clojure-with-core-dot-matrix/
  [& args]
  (println "Implementing a neural network based on the tutorial at: http://gigasquidsoftware.com/blog/2013/12/02/neural-networks-in-clojure-with-core-dot-matrix/")


  (println (layer-activation input-neurons input-hidden-strengths))
  (println (layer-activation new-hidden-neurons hidden-output-strengths))
  (println (output-deltas targets new-output-neurons))

  )
