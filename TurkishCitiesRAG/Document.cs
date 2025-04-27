class Document
{
    public string Title { get; set; }
    public string Content { get; set; }
    public float[] Embedding { get; set; }

    public Document(string title, string content)
    {
        Title = title;
        Content = content;
        Embedding = [];
    }
}
