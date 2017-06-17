require 'csv'
require 'liblinear'

x_data = []
y_data = []
# Load data from CSV file into two arrays - one for independent variables X and one for the dependent variable Y
CSV.foreach("./data/admission.csv", :headers => false) do |row|
  x_data.push( [row[0].to_f, row[0].to_f**2, row[1].to_f, row[1].to_f**2, row[1].to_f * row[0].to_f] )
  #x_data.push( [row[0].to_f, row[1].to_f] )
  y_data.push( row[2].to_i )
end

# Divide data into a training set and test set
test_size_percentange = 20.0 # 20.0%
test_set_size = x_data.size * (test_size_percentange/100.to_f)

test_x_data = x_data[0 .. (test_set_size-1)]
test_y_data = y_data[0 .. (test_set_size-1)]

training_x_data = x_data[test_set_size .. x_data.size]
training_y_data = y_data[test_set_size .. y_data.size]

# Setup model and train using training data
model = Liblinear.train(
  { solver_type: Liblinear::L2R_LR },   # Solver type: L2R_LR - L2-regularized logistic regression
  training_y_data,                      # Training data classification
  training_x_data,                      # Training data independent variables
  100                                   # Bias
)

# Predict class
prediction = Liblinear.predict(model, [45, 45**2, 85, 85**2, 45*85])
#prediction = Liblinear.predict(model, [45, 85])
# get prediction probablities
probs = Liblinear.predict_probabilities(model, [45, 45**2, 85, 85**2, 45*85])
#probs = Liblinear.predict_probabilities(model, [45, 85])
probs = probs.sort

puts "Algorithm predicted class #{prediction}"
puts "#{(probs[1]*100).round(2)}% probablity of prediction"
puts "#{(probs[0]*100).round(2)}% probablity of being other class"


predicted = []
test_x_data.each do |params|
  predicted.push( Liblinear.predict(model, params) )
end
correct = predicted.collect.with_index { |e,i| (e == test_y_data[i]) ? 1 : 0 }.inject{ |sum,e| sum+e }

puts "Accuracy: #{((correct.to_f / test_set_size) * 100).round(2)}% - test set of size #{test_size_percentange}%"
