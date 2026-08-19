import { useState } from "react";
import {
    View,
    Text,
    TouchableOpacity,
    Image,
    ScrollView,
    ActivityIndicator,
    StyleSheet,
    Platform,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { Image as ExpoImage } from "expo-image";

const SERVER_URL = "https://jeffreygoodgood-pokemon-card-scanner.hf.space";

// Type definitions for the server response
interface Attack {
    name: string;
    cost: string[];
    damage: number | null;
    effect: string | null;
}

interface Ability {
    name: string;
    type: string;
    effect: string | null;
}

interface TCGPlayerVariant {
    lowPrice: number | null;
    midPrice: number | null;
    highPrice: number | null;
    marketPrice: number | null;
    directLowPrice: number | null;
}

interface TCGPlayerPricing {
    unit: string;
    normal?: TCGPlayerVariant | null;
    reverse?: TCGPlayerVariant | null;
    [key: string]: unknown;
}

interface CardmarketPricing {
    unit: string;
    avg: number | null;
    low: number | null;
    trend: number | null;
    [key: string]: unknown;
}

interface CardMatch {
    card_id: string;
    distance: number;
    name: string;
    hp?: number;
    types?: string[];
    rarity?: string;
    category?: string;
    stage?: string;
    illustrator?: string;
    retreat?: number;
    set?: { id: string; name: string };
    image?: string;
    attacks?: Attack[];
    abilities?: Ability[];
    pricing?: {
        tcgplayer?: TCGPlayerPricing | null;
        cardmarket?: CardmarketPricing | null;
    };
}

interface Top5Card {
    card_id: string;
    name: string;
    image: string | null;
    distance: number;
}

interface IdentifyResponse {
    success: boolean;
    error?: string;
    match?: CardMatch;
    top5?: Top5Card[];
}

export default function ScanScreen() {
    const [photo, setPhoto] = useState<string | null>(null);
    const [result, setResult] = useState<IdentifyResponse | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [loadingAlt, setLoadingAlt] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [showAlternatives, setShowAlternatives] = useState<boolean>(false);

    const takePhoto = async () => {
        const permission = await ImagePicker.requestCameraPermissionsAsync();
        if (!permission.granted) {
            setError("Camera permission is required");
            return;
        }

        const pickerResult = await ImagePicker.launchCameraAsync({
            quality: 0.8,
            allowsEditing: false,
        });

        if (!pickerResult.canceled) {
            const asset = pickerResult.assets[0];
            setPhoto(asset.uri);
            setResult(null);
            setError(null);
            setShowAlternatives(false);
            identifyCard(asset.uri);
        }
    };

    const pickFromGallery = async () => {
        const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (!permission.granted) {
            setError("Gallery permission is required");
            return;
        }

        const pickerResult = await ImagePicker.launchImageLibraryAsync({
            quality: 0.8,
            allowsEditing: false,
        });

        if (!pickerResult.canceled) {
            const asset = pickerResult.assets[0];
            setPhoto(asset.uri);
            setResult(null);
            setError(null);
            setShowAlternatives(false);
            identifyCard(asset.uri);
        }
    };

    const identifyCard = async (imageUri: string) => {
        setLoading(true);
        setError(null);

        try {
            const formData = new FormData();

            if (Platform.OS === "web") {
                const res = await fetch(imageUri);
                const blob = await res.blob();
                formData.append("image", blob, "card.jpg");
            } else {
                formData.append("image", {
                    uri: imageUri,
                    type: "image/jpeg",
                    name: "card.jpg",
                } as unknown as Blob);
            }

            const response = await fetch(`${SERVER_URL}/identify`, {
                method: "POST",
                body: formData,
            });

            const data: IdentifyResponse = await response.json();

            if (data.success) {
                setResult(data);
            } else {
                setError(data.error || "Could not identify card");
            }
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : "Unknown error";
            setError(`Connection error: ${message}\n\nMake sure the server is running at ${SERVER_URL}`);
        } finally {
            setLoading(false);
        }
    };

    const selectAlternative = async (card: Top5Card) => {
        setLoadingAlt(true);
        try {
            // Fetch full metadata for the selected alternative from the server
            const response = await fetch(`${SERVER_URL}/card/${card.card_id}`);

            const metadata: CardMatch = await response.json();

            // Swap the match with the selected alternative
            setResult((prev) => {
                if (!prev) return prev;
                return {
                    ...prev,
                    match: {
                        ...metadata,
                        card_id: card.card_id,
                        distance: card.distance,
                    },
                };
            });
            setShowAlternatives(false);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : "Unknown error";
            setError(`Failed to load card details: ${message}`);
        } finally {
            setLoadingAlt(false);
        }
    };

    // Helper to get the best available pricing
    const renderPricing = (match: CardMatch) => {
        if (!match.pricing) return null;

        // Prefer TCGPlayer (USD) over Cardmarket (EUR)
        const tcg = match.pricing.tcgplayer;
        const cm = match.pricing.cardmarket;

        // TCGPlayer: data is under normal/reverse variants
        if (tcg) {
            const variant = tcg.normal || tcg.reverse;
            if (variant && variant.marketPrice) {
                return (
                    <View style={styles.attacksSection}>
                        <Text style={styles.sectionTitle}>Market Price (USD)</Text>
                        <View style={styles.attackItem}>
                            <Text style={styles.priceText}>
                                Market: ${variant.marketPrice.toFixed(2)}
                            </Text>
                            {variant.lowPrice && (
                                <Text style={styles.attackCost}>Low: ${variant.lowPrice.toFixed(2)}</Text>
                            )}
                            {variant.midPrice && (
                                <Text style={styles.attackCost}>Mid: ${variant.midPrice.toFixed(2)}</Text>
                            )}
                            {variant.highPrice && (
                                <Text style={styles.attackCost}>High: ${variant.highPrice.toFixed(2)}</Text>
                            )}
                        </View>
                    </View>
                );
            }
        }

        // Fallback: Cardmarket (EUR)
        if (cm && cm.avg) {
            return (
                <View style={styles.attacksSection}>
                    <Text style={styles.sectionTitle}>Market Price (EUR)</Text>
                    <View style={styles.attackItem}>
                        <Text style={styles.priceText}>Avg: €{cm.avg.toFixed(2)}</Text>
                        {cm.low && <Text style={styles.attackCost}>Low: €{cm.low.toFixed(2)}</Text>}
                        {cm.trend && <Text style={styles.attackCost}>Trend: €{cm.trend.toFixed(2)}</Text>}
                    </View>
                </View>
            );
        }

        return null;
    };

    return (
        <ScrollView style={styles.container} contentContainerStyle={styles.content}>
            <Text style={styles.title}>Pokémon Card Scanner</Text>

            {/* Buttons */}
            <View style={styles.buttonRow}>
                <TouchableOpacity style={styles.button} onPress={takePhoto}>
                    <Text style={styles.buttonText}>Take Photo</Text>
                </TouchableOpacity>
                <TouchableOpacity style={styles.buttonSecondary} onPress={pickFromGallery}>
                    <Text style={styles.buttonSecondaryText}>Gallery</Text>
                </TouchableOpacity>
            </View>

            {/* Photo preview */}
            {photo && (
                <Image source={{ uri: photo }} style={styles.preview} resizeMode="contain" />
            )}

            {/* Loading indicator */}
            {loading && (
                <View style={styles.loadingContainer}>
                    <ActivityIndicator size="large" color="#E3350D" />
                    <Text style={styles.loadingText}>Identifying card...</Text>
                </View>
            )}

            {/* Error message */}
            {error && (
                <View style={styles.errorContainer}>
                    <Text style={styles.errorText}>{error}</Text>
                </View>
            )}

            {/* Results */}
            {result && result.match && (
                <View style={styles.resultContainer}>
                    <Text style={styles.resultTitle}>Match Found!</Text>

                    {/* Card image from TCGDex */}
                    {result.match.image && (
                        <ExpoImage
                            source={{ uri: `${result.match.image}/high.png` }}
                            style={styles.cardImage}
                            contentFit="contain"
                            cachePolicy="memory-disk"
                            transition={200}
                        />
                    )}

                    {/* Card details */}
                    <View style={styles.detailsContainer}>
                        <Text style={styles.cardName}>{result.match.name}</Text>
                        <Text style={styles.cardId}>{result.match.card_id}</Text>

                        {result.match.hp && (
                            <Text style={styles.detail}>HP: {result.match.hp}</Text>
                        )}
                        {result.match.types && (
                            <Text style={styles.detail}>Type: {result.match.types.join(", ")}</Text>
                        )}
                        {result.match.rarity && (
                            <Text style={styles.detail}>Rarity: {result.match.rarity}</Text>
                        )}
                        {result.match.set && (
                            <Text style={styles.detail}>
                                Set: {result.match.set.name} ({result.match.set.id})
                            </Text>
                        )}
                        {result.match.category && (
                            <Text style={styles.detail}>Category: {result.match.category}</Text>
                        )}
                        {result.match.stage && (
                            <Text style={styles.detail}>Stage: {result.match.stage}</Text>
                        )}
                        {result.match.retreat !== undefined && (
                            <Text style={styles.detail}>Retreat Cost: {result.match.retreat}</Text>
                        )}
                        {result.match.illustrator && (
                            <Text style={styles.detail}>Illustrator: {result.match.illustrator}</Text>
                        )}

                        {/* Pricing — USD preferred, EUR fallback */}
                        {renderPricing(result.match)}

                        {/* Attacks */}
                        {result.match.attacks && result.match.attacks.length > 0 && (
                            <View style={styles.attacksSection}>
                                <Text style={styles.sectionTitle}>Attacks</Text>
                                {result.match.attacks.map((attack, idx) => (
                                    <View key={idx} style={styles.attackItem}>
                                        <Text style={styles.attackName}>
                                            {attack.name} {attack.damage ? `— ${attack.damage}` : ""}
                                        </Text>
                                        {attack.effect && (
                                            <Text style={styles.attackEffect}>{attack.effect}</Text>
                                        )}
                                        {attack.cost && (
                                            <Text style={styles.attackCost}>
                                                Cost: {attack.cost.join(", ")}
                                            </Text>
                                        )}
                                    </View>
                                ))}
                            </View>
                        )}

                        {/* Abilities */}
                        {result.match.abilities && result.match.abilities.length > 0 && (
                            <View style={styles.attacksSection}>
                                <Text style={styles.sectionTitle}>Abilities</Text>
                                {result.match.abilities.map((ability, idx) => (
                                    <View key={idx} style={styles.attackItem}>
                                        <Text style={styles.attackName}>{ability.name}</Text>
                                        {ability.effect && (
                                            <Text style={styles.attackEffect}>{ability.effect}</Text>
                                        )}
                                    </View>
                                ))}
                            </View>
                        )}

                        {/* Confidence */}
                        <Text style={styles.distance}>
                            Confidence: {(2 - result.match.distance).toFixed(2)} / 2.00
                        </Text>
                    </View>

                    {/* Not your card? */}
                    {result.top5 && result.top5.length > 1 && !showAlternatives && (
                        <TouchableOpacity
                            style={styles.notYourCardButton}
                            onPress={() => setShowAlternatives(true)}
                        >
                            <Text style={styles.notYourCardText}>Not your card?</Text>
                        </TouchableOpacity>
                    )}

                    {/* Alternative cards — 2x2 grid */}
                    {showAlternatives && result.top5 && (
                        <View style={styles.alternativesContainer}>
                            <Text style={styles.sectionTitle}>Select the correct card</Text>

                            {loadingAlt && (
                                <View style={styles.loadingContainer}>
                                    <ActivityIndicator size="large" color="#E3350D" />
                                    <Text style={styles.loadingText}>Loading card details...</Text>
                                </View>
                            )}

                            <View style={styles.alternativesGrid}>
                                {result.top5.slice(1).map((card, idx) => (
                                    <TouchableOpacity
                                        key={idx}
                                        style={styles.alternativeCard}
                                        onPress={() => selectAlternative(card)}
                                    >
                                        {card.image ? (
                                            <ExpoImage
                                                source={{ uri: `${card.image}/low.png` }}
                                                style={styles.alternativeImage}
                                                contentFit="contain"
                                                cachePolicy="memory-disk"
                                                transition={200}
                                            />
                                        ) : (
                                            <View style={styles.alternativePlaceholder}>
                                                <Text style={styles.alternativePlaceholderText}>
                                                    {card.name}
                                                </Text>
                                            </View>
                                        )}
                                    </TouchableOpacity>
                                ))}
                            </View>

                            <TouchableOpacity
                                style={styles.cancelButton}
                                onPress={() => setShowAlternatives(false)}
                            >
                                <Text style={styles.cancelButtonText}>Cancel</Text>
                            </TouchableOpacity>
                        </View>
                    )}
                </View>
            )}
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: "#1a1a2e",
    },
    content: {
        padding: 20,
        paddingTop: 60,
        alignItems: "center",
    },
    title: {
        fontSize: 28,
        fontWeight: "bold",
        color: "#ffffff",
        marginBottom: 20,
    },
    buttonRow: {
        flexDirection: "row",
        gap: 12,
        marginBottom: 20,
    },
    button: {
        backgroundColor: "#E3350D",
        paddingVertical: 14,
        paddingHorizontal: 28,
        borderRadius: 12,
    },
    buttonText: {
        color: "#ffffff",
        fontSize: 18,
        fontWeight: "bold",
    },
    buttonSecondary: {
        backgroundColor: "transparent",
        paddingVertical: 14,
        paddingHorizontal: 28,
        borderRadius: 12,
        borderWidth: 2,
        borderColor: "#E3350D",
    },
    buttonSecondaryText: {
        color: "#E3350D",
        fontSize: 18,
        fontWeight: "bold",
    },
    preview: {
        width: 280,
        height: 380,
        borderRadius: 12,
        marginBottom: 16,
    },
    loadingContainer: {
        alignItems: "center",
        marginVertical: 20,
    },
    loadingText: {
        color: "#cccccc",
        marginTop: 10,
        fontSize: 16,
    },
    errorContainer: {
        backgroundColor: "#3a1a1a",
        padding: 16,
        borderRadius: 12,
        marginVertical: 10,
        width: "100%",
    },
    errorText: {
        color: "#ff6b6b",
        fontSize: 14,
        textAlign: "center",
    },
    resultContainer: {
        width: "100%",
        alignItems: "center",
        marginTop: 10,
    },
    resultTitle: {
        fontSize: 22,
        fontWeight: "bold",
        color: "#4ecca3",
        marginBottom: 12,
    },
    cardImage: {
        width: 260,
        height: 360,
        borderRadius: 12,
        marginBottom: 16,
    },
    detailsContainer: {
        width: "100%",
        backgroundColor: "#16213e",
        borderRadius: 12,
        padding: 16,
        marginBottom: 12,
    },
    cardName: {
        fontSize: 24,
        fontWeight: "bold",
        color: "#ffffff",
        marginBottom: 4,
    },
    cardId: {
        fontSize: 14,
        color: "#888888",
        marginBottom: 12,
    },
    detail: {
        fontSize: 16,
        color: "#cccccc",
        marginBottom: 4,
    },
    attacksSection: {
        marginTop: 12,
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: "bold",
        color: "#E3350D",
        marginBottom: 8,
    },
    attackItem: {
        backgroundColor: "#1a1a3e",
        padding: 10,
        borderRadius: 8,
        marginBottom: 6,
    },
    attackName: {
        fontSize: 16,
        fontWeight: "bold",
        color: "#ffffff",
    },
    attackEffect: {
        fontSize: 13,
        color: "#aaaaaa",
        marginTop: 4,
    },
    attackCost: {
        fontSize: 13,
        color: "#888888",
        marginTop: 2,
    },
    priceText: {
        fontSize: 16,
        fontWeight: "bold",
        color: "#4ecca3",
    },
    distance: {
        fontSize: 14,
        color: "#4ecca3",
        marginTop: 12,
        textAlign: "center",
    },
    notYourCardButton: {
        marginTop: 8,
        marginBottom: 20,
        paddingVertical: 12,
        paddingHorizontal: 24,
        borderRadius: 10,
        borderWidth: 1,
        borderColor: "#888888",
    },
    notYourCardText: {
        color: "#888888",
        fontSize: 16,
    },
    alternativesContainer: {
        width: "100%",
        backgroundColor: "#16213e",
        borderRadius: 12,
        padding: 16,
        marginBottom: 30,
        alignItems: "center",
    },
    alternativesGrid: {
        flexDirection: "row",
        flexWrap: "wrap",
        justifyContent: "center",
        gap: 10,
        marginTop: 8,
    },
    alternativeCard: {
        width: "47%",
        aspectRatio: 0.72,
        borderRadius: 10,
        overflow: "hidden",
        borderWidth: 2,
        borderColor: "transparent",
    },
    alternativeImage: {
        width: "100%",
        height: "100%",
    },
    alternativePlaceholder: {
        width: "100%",
        height: "100%",
        backgroundColor: "#1a1a3e",
        justifyContent: "center",
        alignItems: "center",
        padding: 8,
    },
    alternativePlaceholderText: {
        color: "#cccccc",
        fontSize: 14,
        textAlign: "center",
    },
    cancelButton: {
        marginTop: 12,
        paddingVertical: 10,
        paddingHorizontal: 20,
        borderRadius: 8,
        borderWidth: 1,
        borderColor: "#E3350D",
    },
    cancelButtonText: {
        color: "#E3350D",
        fontSize: 16,
    },
});