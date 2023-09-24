At the moment:
- Our app usually (though not always) returns the right property - and seems to always get the details of the property right.
- BUT if there are multiple properties that it should return, it doesn't.

These two problems make me think it *might* be a problem with the way the DataFrame is getting split up, and turned into embeddings. 
- So we may want to take this into our own hands a bit.

The multiple properties thing might be a bit harder than this, though - I'm not sure if the semantic similarity part of this only returns one property.
- This is worth diving into.
- See if others have a similar problem, where they want *multiple* responses from their RetrievalQA chain. I think I saw something vaguely like this in the docs.

SO:
- probably do that last thing (trying to get multiple repsonses). This will also probably necessitate getting funky with the embeddings,

THEN:
- Write a bunch of tests, basically.