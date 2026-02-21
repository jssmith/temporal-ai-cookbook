Provide a few sentences of introduction that will make sense to someone who doesn’t know what these algorithms are / what they do? this will make the material more approachable and improve and SEO + GEO
Make it run faster on my MacBook pro - can we parametrize the size of the sweep and make sure. it should complete in a few minutes without draining my batter
README Should explain what is goin on wih the code
Provide a diagram to illustrate the process <- likely needed to make this good
Tell me when I should be hitting ctrl+c to see the logs i find myself getting
Hard to understand what is happening in the Temporal UI - so many workflows, where do I look first?

Can we consolidate this into a single readme. do we really want competitive analysis? this seems like a moving target.

Pegging my GPU and I don’t have a clear sense of progress

Where are checkpoints stored?
Scaling out: shared file system needs to be provided - how do we do that in cloud environment? Lustre?

Where was this adapted from? what is the original example

How can I monitor the loss as the fine-tuning job proceeds on the loss as the fine tuning proceeds?


confusing logs: i see a lot of output like this, and I can't tell whether it is working as intended
Seeing lots of   Key                                     | Status     |
----------------------------------------+------------+-
mask_predictions.LayerNorm.bias         | UNEXPECTED |
lm_predictions.lm_head.dense.bias       | UNEXPECTED |
lm_predictions.lm_head.LayerNorm.weight | UNEXPECTED |
mask_predictions.classifier.weight      | UNEXPECTED |
mask_predictions.dense.weight           | UNEXPECTED |
lm_predictions.lm_head.LayerNorm.bias   | UNEXPECTED |
mask_predictions.classifier.bias        | UNEXPECTED |
mask_predictions.dense.bias             | UNEXPECTED |
mask_predictions.LayerNorm.weight       | UNEXPECTED |
lm_predictions.lm_head.dense.weight     | UNEXPECTED |
lm_predictions.lm_head.bias             | UNEXPECTED |
pooler.dense.weight                     | MISSING    |
pooler.dense.bias                       | MISSING    |
classifier.bias                         | MISSING    |
classifier.weight                       | MISSING    |

