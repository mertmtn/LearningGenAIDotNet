using Microsoft.Extensions.AI;


var knowledgeBase = GetKnowledgeBaseFromDataSource();

// Başlangıçta her dokümana embedding hesapla
Console.WriteLine("Knowledge base reading...");
foreach (var doc in knowledgeBase)
{
    doc.Embedding = await GetEmbedding(doc.Content);
}
Console.WriteLine("Knowledge base ready");

// Kullanıcıdan soru al
Console.Write("Question: ");
var userQuestion = Console.ReadLine() ?? "";
var userEmbedding = await GetEmbedding(userQuestion);

// En benzer dokümanı bul
var bestMatch = knowledgeBase
    .Select(doc => new
    {
        Document = doc,
        Score = CosineSimilarity(userEmbedding, doc.Embedding)
    })
    .OrderByDescending(x => x.Score)
    .FirstOrDefault();

if (bestMatch == null)
{
    Console.WriteLine("Not found.");
    return;
}

// Cevap için yeni bir prompt hazırla
var finalPrompt = $"Aşağıdaki bilgiye dayanarak soruyu cevapla:\n{bestMatch.Document.Content}\nSoru: {userQuestion}";

var answer = await AskLLM(finalPrompt);
Console.WriteLine(answer);

async Task<float[]> GetEmbedding(string text)
{
    var generatorOpenAi = new OpenAI.Embeddings.EmbeddingClient("text-embedding-3-small", "YOUR_API_KEY")
                                                .AsIEmbeddingGenerator();

    var generator = new OllamaEmbeddingGenerator("http://localhost:11434/", "all-minilm");

    var embedding = await generator.GenerateEmbeddingAsync(text);
    var embeddingArray = embedding.Vector.ToArray();

    return embeddingArray;
}

async Task<ChatResponse> AskLLM(string prompt)
{
    var openAiClient = new OpenAI.Chat.ChatClient("gpt-4o-mini", "YOUR_API_KEY").AsIChatClient();
    var client = new OllamaChatClient("http://localhost:11434/", "llama3.1");

    return await client.GetResponseAsync(prompt.ToString());
}

float CosineSimilarity(float[] vectorA, float[] vectorB)
{
    var dotProduct = vectorA.Zip(vectorB, (a, b) => a * b).Sum();
    var magnitudeA = Math.Sqrt(vectorA.Sum(x => x * x));
    var magnitudeB = Math.Sqrt(vectorB.Sum(x => x * x));

    return (float)(dotProduct / (magnitudeA * magnitudeB));
}


static List<Document> GetKnowledgeBaseFromDataSource()
{
    return
[
new("Adana", "Adana, Türkiye'nin güneyinde yer alan ve kebabıyla ünlü bir şehirdir."),
    new ("Adıyaman", "Adıyaman, Nemrut Dağı'ndaki dev heykelleriyle tanınır."),
    new ("Afyonkarahisar", "Afyonkarahisar, termal kaplıcaları ve kaymağı ile meşhurdur."),
    new ("Ağrı", "Ağrı, Türkiye'nin en yüksek dağı olan Ağrı Dağı'na ev sahipliği yapar."),
    new ("Aksaray", "Aksaray, Ihlara Vadisi ve tarihi güzellikleriyle bilinir."),
    new ("Amasya", "Amasya, tarihi kral mezarları ve şehzadeler şehri olarak tanınır."),
    new ("Ankara", "Ankara, Türkiye'nin başkenti ve siyasi merkezidir."),
    new ("Antalya", "Antalya, turistik sahilleri ve tarihi kalıntılarıyla ünlüdür."),
    new ("Ardahan", "Ardahan, serin iklimi ve doğal güzellikleriyle bilinir."),
    new ("Artvin", "Artvin, Karadeniz'in yemyeşil doğasına ve yaylalarına sahiptir."),
    new ("Aydın", "Aydın, tarihi Efes Antik Kenti'ne yakınlığıyla tanınır."),
    new ("Balıkesir", "Balıkesir, hem Marmara hem Ege Denizi'ne kıyısı olan bir şehirdir."),
    new ("Bartın", "Bartın, Amasra gibi tarihi ve turistik ilçelere sahiptir."),
    new ("Batman", "Batman, Hasankeyf antik kentiyle tanınan Güneydoğu Anadolu şehridir."),
    new ("Bayburt", "Bayburt, Türkiye'nin en küçük illerinden biridir."),
    new ("Bilecik", "Bilecik, Osmanlı İmparatorluğu'nun kurulduğu topraklardandır."),
    new ("Bingöl", "Bingöl, çok sayıda göl ve doğal güzelliğe sahiptir."),
    new ("Bitlis", "Bitlis, tarihi kaleleri ve Nemrut Krater Gölü ile meşhurdur."),
    new ("Bolu", "Bolu, Abant Gölü ve Yedigöller Milli Parkı ile ünlüdür."),
    new ("Burdur", "Burdur, Salda Gölü gibi doğal güzelliklere ev sahipliği yapar."),
    new ("Bursa", "Bursa, Osmanlı'nın ilk başkenti ve yeşil doğasıyla tanınır."),
    new ("Çanakkale", "Çanakkale, tarihi Gelibolu Yarımadası ve Truva Antik Kenti ile bilinir."),
    new ("Çankırı", "Çankırı, tuz mağaraları ve kaya mezarlarıyla tanınır."),
    new ("Çorum", "Çorum, tarihi Hitit uygarlığının izlerini taşır."),
    new ("Denizli", "Denizli, Pamukkale Travertenleri ile dünyaca ünlüdür."),
    new ("Diyarbakır", "Diyarbakır, tarihi surları ve kültürel zenginlikleriyle bilinir."),
    new ("Düzce", "Düzce, Karadeniz kıyısında doğal güzellikleriyle dikkat çeker."),
    new ("Edirne", "Edirne, Selimiye Camii ve tarihi köprüleriyle tanınır."),
    new ("Elazığ", "Elazığ, Harput Kalesi ve Hazar Gölü ile meşhurdur."),
    new ("Erzincan", "Erzincan, doğal güzellikleri ve Kemaliye Kanyonu ile bilinir."),
    new ("Erzurum", "Erzurum, kış sporları ve tarihi yapılarıyla ünlüdür."),
    new ("Eskişehir", "Eskişehir, öğrenci şehri olarak kültürel ve sanatsal etkinlikleriyle öne çıkar."),
    new ("Gaziantep", "Gaziantep, mutfağıyla ve tarihi hanlarıyla ünlü bir şehirdir."),
    new ("Giresun", "Giresun, Karadeniz kıyısındaki fındık üretimiyle bilinir."),
    new ("Gümüşhane", "Gümüşhane, tarihi kaleleri ve doğal mağaralarıyla tanınır."),
    new ("Hakkari", "Hakkari, dağlık coğrafyası ve doğal güzellikleriyle dikkat çeker."),
    new ("Hatay", "Hatay, çok kültürlü yapısı ve mutfağıyla bilinir."),
    new ("Iğdır", "Iğdır, Ağrı Dağı'nın eteğinde yer alır."),
    new ("Isparta", "Isparta, gül bahçeleri ve lavanta tarlalarıyla ünlüdür."),
    new ("İstanbul", "İstanbul, hem Avrupa hem Asya kıtalarına yayılan tarihi bir mega kenttir."),
    new ("İzmir", "İzmir, Ege'nin incisi olarak bilinen modern ve turistik bir şehirdir."),
    new ("Kahramanmaraş", "Kahramanmaraş, dondurması ve zengin kültürüyle meşhurdur."),
    new ("Karabük", "Karabük, Safranbolu evleriyle UNESCO Dünya Mirası listesindedir."),
    new ("Karaman", "Karaman, Türk Dil Kurumu'nun kurulduğu şehirdir."),
    new ("Kars", "Kars, Ani Harabeleri ve kış turizmiyle ünlüdür."),
    new ("Kastamonu", "Kastamonu, tarihi evleri ve doğasıyla bilinir."),
    new ("Kayseri", "Kayseri, Erciyes Dağı ve pastırmasıyla tanınır."),
    new ("Kırıkkale", "Kırıkkale, Türkiye'nin önemli sanayi şehirlerinden biridir."),
    new ("Kırklareli", "Kırklareli, Trakya bölgesinde üzüm bağlarıyla meşhurdur."),
    new ("Kırşehir", "Kırşehir, müzik ve kültür şehri olarak bilinir."),
    new ("Kilis", "Kilis, zeytinyağı üretimi ve tarihi yapılarıyla tanınır."),
    new ("Kocaeli", "Kocaeli, sanayi ve liman kenti olarak Marmara'nın önemli şehirlerindendir."),
    new ("Konya", "Konya, Mevlana'nın şehri ve Türkiye'nin yüzölçümü en büyük ilidir."),
    new ("Kütahya", "Kütahya, çinileriyle ünlü tarihi bir şehirdir."),
    new ("Malatya", "Malatya, kayısısıyla dünyaca tanınır."),
    new ("Manisa", "Manisa, üzüm bağları ve tarihi kalıntılarıyla ünlüdür."),
    new ("Mardin", "Mardin, taş mimarisi ve tarihi dokusuyla büyüleyicidir."),
    new ("Mersin", "Mersin, Akdeniz kıyısında büyük bir liman şehridir."),
    new ("Muğla", "Muğla, Bodrum, Marmaris gibi turistik ilçeleriyle ünlüdür."),
    new ("Muş", "Muş, doğal güzellikleri ve lalesiyle bilinir."),
    new ("Nevşehir", "Nevşehir, Kapadokya bölgesi ve peri bacalarıyla ünlüdür."),
    new ("Niğde", "Niğde, tarihi yeraltı şehirleriyle tanınır."),
    new ("Ordu", "Ordu, Karadeniz'in sahil şehri olup fındık üretimiyle öne çıkar."),
    new ("Osmaniye", "Osmaniye, doğal güzellikleri ve tarihi kaleleriyle bilinir."),
    new ("Rize", "Rize, çayı ve yemyeşil yaylalarıyla tanınır."),
    new ("Sakarya", "Sakarya, doğal güzellikleri ve Sapanca Gölü ile ünlüdür."),
    new ("Samsun", "Samsun, Karadeniz Bölgesi'nin en büyük şehirlerinden biridir."),
    new ("Siirt", "Siirt, büryan kebabı ve doğal güzellikleriyle tanınır."),
    new ("Sinop", "Sinop, Türkiye'nin en kuzeydeki şehri ve tarihi kaleleriyle bilinir."),
    new ("Sivas", "Sivas, tarihi kongresi ve doğal termal kaynaklarıyla bilinir."),
    new ("Şanlıurfa", "Şanlıurfa, Göbeklitepe gibi insanlık tarihine ışık tutan yerlere sahiptir."),
    new ("Şırnak", "Şırnak, dağlık doğası ve kültürel çeşitliliğiyle dikkat çeker."),
    new ("Tekirdağ", "Tekirdağ, Marmara Bölgesi'nde şarap üretimiyle tanınır."),
    new ("Tokat", "Tokat, tarihi kaleleri ve doğal güzellikleriyle meşhurdur."),
    new ("Trabzon", "Trabzon, Sümela Manastırı ve Karadeniz kültürüyle bilinir."),
    new ("Tunceli", "Tunceli, Munzur Dağları ve doğa sporlarıyla tanınır."),
    new ("Uşak", "Uşak, tarihi antik kentleri ve halılarıyla bilinir."),
    new ("Van", "Van, Van Gölü ve kedileriyle meşhurdur."),
    new ("Yalova", "Yalova, termal kaplıcaları ve sahil şeridiyle dikkat çeker."),
    new ("Yozgat", "Yozgat, İç Anadolu'nun tarihi ve doğal güzellikleriyle öne çıkar."),
    new ("Zonguldak", "Zonguldak, Türkiye'nin ilk maden şehirlerinden biridir.")
];
}
