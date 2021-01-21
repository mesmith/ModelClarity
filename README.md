# ModelClarity
Machine learning with pytorch, with an emphasis on model clarity.

This is a simple neural network model in pandas and pytorch. It takes an arbitrary dataset consisting of categorical and continuous variables, and allows us to designate one column as the label, and a set of other columns as features that may or may not predict the labels.

The takeaway from this model is to answer a fairly common objection that python models in general, and pytorch models specifically, fundamentally lack transparency.

This has negative effects in the sense that data science developer labor--not training efficiency, nor production efficiency--is typically the most expensive component in any machine learning project.

I believe that this labor cost can be vastly improved by:

* better documentation within models, whereby the data scientist stops assuming that there are no consumers of the program text; and

* a bias towards assuming that the next reader of the program text is not expert in all of the details of pandas, numpy, tensors, and pytorch.

The goal of this model is to demonstrate that it is possible to write a pytorch model that a reader with a modest understanding of machine learning can understand without having to refer to online searches, courseware, or face to face conversation.

Ideally, you can use this model as a machine learning workbench. I'll call it "knowledge transfer learning", where hopefully you can convert your knowledge of machine learning into learning how pytorch works.

Notably, this model splits the data into the "holy trinity" of training, cross validation, and testing data, as well as calculating and displaying losses, accuracy, precision, recall, and
F1* scores.
