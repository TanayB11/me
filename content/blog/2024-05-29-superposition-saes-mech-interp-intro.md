+++
title = "Superposition & SAEs in Mechanistic Interpretability: An Intro"
date = 2024-05-29
description = "🌳"
+++
Anthropic's [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) and Golden Gate Claude are some really exciting work in mechanistic interpretability, so I've been trying to better wrap my head around them. I wasn't satisfied by existing explanations of their work, so here's my attempt at something more digestible than the original papers.

> I also have some (work-in-progress) [code](https://github.com/TanayB11/sae-interp/tree/main/toys) to show for it!

## Features & Superposition
Anthropic's Transformer Circuits Thread has had a [few](https://transformer-circuits.pub/2022/toy_model/index.html) previous [works](https://transformer-circuits.pub/2023/monosemantic-features/index.html) that try to interpret the hidden states of deep learning models. Ultimately, the goal is to find meaningful features in models' activations.

But what is a **feature**? We can go by multiple definitions, such as:
1. Aspects of the input data that are meaningful to us humans
2. Directions in the activation space of the neural network's layers

Suppose we have an activation vector <small>$a \in \mathbb{R}^2$</small>, and we're trying to represent two features (according to Definition 1 above)—say "airplane" and "water bottle." Here, we have 2 features and 2 dimensions, so we're able to assign orthogonal basis vectors to each feature.

I trained a toy 1-layer model, following this [notebook](https://colab.research.google.com/drive/15S4ISFVMQtfc0FPi29HRaX03dWxL65zx?usp=sharing). More details on that <a href='#details-on-the-toy-model'>later</a>.

For now, here are the actual results ([code](https://github.com/TanayB11/sae-interp/blob/main/toys/bottleneck_models.py)):

<br/>

<center>
    <img src="/images/2024-05-29-superposition-saes-mech-interp-intro/2feats.png" width="60%"
    style="border-radius: 0.5em;"/>
</center>


Note how the features that this toy model has learned don't exactly line up with the standard basis—there's no reason they have to! That's because the models I trained have a [non-privileged basis](https://transformer-circuits.pub/2022/toy_model/index.html#motivation-privileged). I can rotate all of the features by some arbitrary rotation matrix <small>$R$</small>, and the model will behave in exactly the same way as it does (provided I also rotate the inputs by <small>$R$</small>).

<center>
    <img src="/images/2024-05-29-superposition-saes-mech-interp-intro/privileged_basis.png" width="90%"
    style="border-radius: 0.05em;"/>
</center>

> Source: [Anthropic's Toy Models Paper](https://transformer-circuits.pub/2022/toy_model/index.html#motivation-privileged)

<br/>

Now, let's scale things up. Suppose we're instead trying to use represent 3 or 5 features in our activation space <small>$a$</small>. All of a sudden, we [can't](https://openreview.net/pdf?id=F76bwRSLeK) give each feature its own orthogonal basis direction. We have to squeeze 3 or 5 features into <small>$\mathbb{R}^2$</small> somehow—and the best way is to pick directions that have the least overlap/interference (i.e. minimize cosine similarity between any 2 features).

> *NB* 1: There's a beautiful connection to [bond angles](https://transformer-circuits.pub/2022/toy_model/index.html#geometry-structures) in chemistry here!
> 
> *NB* 2: You'll sometimes read that these features form an **overcomplete basis** for the activation space.

This phenomenon is called **superposition**!

<table>
  <tr>
    <td>
        <img src="/images/2024-05-29-superposition-saes-mech-interp-intro/3feats.png" width="100%" style="border-radius: 0.05em;"/>
    </td>
    <td>
        <img src="/images/2024-05-29-superposition-saes-mech-interp-intro/5feats.png" width="100%" style="border-radius: 0.05em;"/>
    </td>
  </tr>
</table>

<br/>

<p id='sparsity_superposition'>
Actually, I missed one thing: The model won't <em>always</em> spread out the vectors like this. It only does this when every feature occurs somewhat infrequently (<b>sparsely</b>) in the training data.
</p>

1. If the features aren't sparse (they occur frequently in the data), the model chooses to just represent only the most important features in the activation space. In this case, we don't get much superposition.
1. If the features are sparse (they occur infrequently in the data), the model will represent the important features in the activation space. If we have enough important sparse features, the model will use superposition to represent all of them with minimum overlap/interference.

<br/>

<center>
    <img src="/images/2024-05-29-superposition-saes-mech-interp-intro/sparsity_superposition.png" width="90%" style="border-radius: 0.05em;"/>
</center>

> Source: [Anthropic's Toy Models Paper](https://transformer-circuits.pub/2022/toy_model/index.html)

<br/>

Here are some intuitive explanations of sparsity and importance:
> <u>Example 1:</u> Say we have a model that predicts annual income. It's trained on features <small>$\{\text{attendedCollege}, \text{occupation}, \text{hasBrownHair}\}$</small>, but only 5% of the dataset examples have brown hair. Since hair color is sparse in this dataset, and it's (probably) not a good indicator of income, then the model will likely send the feature vector representing "hasBrownHair" to zero. Superposition (probably) won't happen.
> 
> <u>Example 2:</u> Say we have a model that predicts colorblindness. It's trained on features <small>$\{\text{age}, \text{citizenship}, \text{isMale}\}$</small>, but the dataset examples are only 5% male. The "male" feature is very sparse—it occurs 5% of the time—but when predicting colorblindness, it's *very* important. So the model will likely *not* send that feature vector to 0 in the activation space. If there are enough other sparse, important features like "isMale," the model will (probably) represent them all in superposition.

<br/>

Of course, in deep neural nets, this all happens in very high-dimensional vector spaces. But the basic principles are the same.

## Details on the Toy Model

The 1-layer model was a simple model that first tries to map its input <small>$x \in \mathbb{R}^{1 \times n}$</small> to </small>$\mathbb{R}^2$</small>, then reconstruct it to get <small>$\hat{x}$</small>. Formally, with our weight matrix <small>$W \in \mathbb{R}^{n \times 2}$</small> and bias <small>$b \in \mathbb{R}^n$</small>,

<small>$$h = xW$$</small>
<small>$$\hat{x} = \text{ReLU}(h W^T + b)$$</small>

We train using the MSE loss (to push the predictions closer to the original inputs).

Now we can extract the features from the toy 1-layer model (relatively) easily. Let's look at <small>$W$</small>. Spelled out, we have

<small>
$$ x = \begin{bmatrix} x_1 & \dots & x_n \end{bmatrix} \quad W = \begin{bmatrix} a_0(x_1) & a_1(x_1)\\ \vdots&\vdots \\ a_0(x_n) & a_1(x_n) \end{bmatrix} $$
</small>

The features are very easy to find in this model: The <small>$i$</small>th row of <small>$W$</small> corresponds to the <small>$i$</small>th feature, <small>$x_i$</small>, embedded in <small>$\mathbb{R}^2$</small>. Thus, the model represents each feature <small>$x_i$</small> as the direction of its activation vector, <small>$a(x_i)$</small>. Our <small>$n$</small> features are then just <small>$\{a_1, \dots, a_n\}$</small>. That's it!

## Motivating Sparse Autoencoders: Sparse Dictionary Learning

With deeper models, this kind of interpretability isn't easy. We can interpret the weight matrix applied to the input, but what if we want to find meaningful features in deep hidden layer? Maybe we're trying to interpret the activations within hidden state of a transformer (the residual stream), or we're trying to interpret the activations of a deep MLP.

Sparse autoencoders (SAEs) are a popular approach, and Anthropic used them in the Scaling Monosemanticity paper to find interpretable features in Claude 3 Sonnet. Let's motivate them from first principles:

Say we want to interpret the activations of an intermediate layer of some deep model <small>$f(x)$</small>.

<center>
    <img src="/images/2024-05-29-superposition-saes-mech-interp-intro/activation.png" width="65%" style="border-radius: 0.05em;"/>
</center>

Our ([substantiated](https://tanaybiradar.com/blog/notes-on-zoom-in-circuits/)) hope is that in each hidden layer, the model is trying to represent a **ton** of features that the model learned during training.

Thus, we're trying to analyze an intermediate activation <small>$a$</small>, which is a <small>$d$</small>-dimensional vector that's trying to represent a **lot** more than <small>$d$</small> features (ex. "traffic cone", "studying in a library", "smokestack", "geologist", etc.). That means we're going to have superposition!

If we're trying to find interpretable features from <small>$a$</small>, then we're trying to find a set of directions in the activation space. Each direction in this activation space should correspond to something that makes sense to us humans.

The question is, <u>how do we figure out a good set of vectors?</u>

Since our activation space doesn't have enough dimensions to give each feature its own orthogonal basis vector, why not map <small>$a$</small> to a "higher-dimensional space" that does?

> We don't know exactly how many features <small>$a$</small> is trying to represent, so we just want to map it to a high-*enough* dimensional space.
> 
> To get really in the weeds: My intuition is that we want to map <small>$a$</small> to a high dimensional space, where it's more likely that each entry in the high-dimensional vector (i.e. each neuron) corresponds to a feature (...at least in a privileged basis, where the standard basis directions are likely more meaningful). We have to be careful about privileged/non-privileged bases when talking about "neurons corresponding to features."

The idea is that <small>$a$</small> is some combination of interpretable features in this "higher-dimensional space," which has then been projected into the lower-dimensional activation space.

Since <small>$a$</small> is representing features in superposition, <a href='#sparsity_superposition'>that means</a> the features occurred sparsely in the training dataset. (For any data example, only a few features were present.) Thus, <small>$a$</small> is a *sparse* combination of these "higher-dimensional features."

So, we want an algorithm that maps <small>$a$</small> to a sparse vector in this "higher-dimensional space."

Sparse autoencoders (SAEs) do exactly that!

## Sparse Autoencoders
The structure of a basic autoencoder is pretty simple. It's just an MLP with one hidden layer. It maps the input activation <small>$a$</small> to a high-dimensional hidden state <small>$h$</small>. Then, using <small>$h$</small>, it computes <small>$\hat{a}$</small>, a reconstruction of <small>$a$</small>.

The intuition is that <small>$h$</small> will contain some useful information about <small>$a$</small>, since it's an intermediate layer in the computation.

<center>
    <img src="/images/2024-05-29-superposition-saes-mech-interp-intro/sae.png" width="30%" style="border-radius: 0.05em;"/>
</center>

In essence, the SAE computation looks like this:

<small>$$h = a^TW_\text{enc}$$</small>
<small>$$\hat{a} = \text{ReLU}(hW_\text{dec})$$</small>

<small>$$a \in \mathbb{R}^d \quad W_\text{enc} \in \mathbb{R}^{d \times m} \quad W_\text{dec} \in \mathbb{R}^{m \times d}$$</small>

In reality, Anthropic made a few [modifications](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-dataset), but the idea is the same.

Since we're trying to reconstruct the original input (the activation <small>$a$</small>), we use an MSE loss. But since <small>$a$</small> is a sparse combination of features in <small>$h$</small>, we also want to ensure that <small>$h$</small> is a sparse vector—so we add an L1 penalty/regularization term (with hyperparameter <small>$\lambda$</small>). For a single training example, our SAE loss is

<small> $$\mathcal{L}(a, \hat{a}) = ||a - \hat{a}||_2^2 + \lambda ||h||_1$$ </small>

> To be super clear, the original model <small>$f(x)$</small> was trained on the <u>input</u> dataset. The SAE is trained on <small>$a$</small>, the <u>activations</u> that were gathered from running the original model on a bunch of data.

There are a few more details (dead neuron resampling, etc.) that improve the utility of the SAEs—but this is the core.

Now that we have features, we can interpret them. For large language models, all we have to do is prompt them carefully and see what features in the SAE activate.

If we input a bunch of prompts related to dolphins to an LLM, collect the activations, run them through the SAE, and see that neuron <small>$i$</small> consistently activates, then we can be reasonably confident that the <small>$i$</small>th neuron of the SAE corresponds to dolphins.

For further reading, I'll point to [this section](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#searching) and [this section](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html#feature-survey) of the Scaling Monosemanticity paper (which are relatively digestible). If you want to read even further into mechanistic interpretability, look into this [paper](https://arxiv.org/abs/2310.01405) and [post](https://vgel.me/posts/) on steering vectors, which are related!

## References
1. [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
2. [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
3. [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
4. [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://openreview.net/pdf?id=F76bwRSLeK)
5. [MATS Colab Exercises](https://colab.research.google.com/drive/15S4ISFVMQtfc0FPi29HRaX03dWxL65zx?usp=sharing)
