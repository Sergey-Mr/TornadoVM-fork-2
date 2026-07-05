package uk.ac.manchester.tornado.examples.compute.custom;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Parser for MCP grid configuration.
 * Parses grid_config from MCP response and evaluates expressions like "size*size".
 */
public class GridConfigParser {

    /**
     * Parsed grid configuration from MCP response.
     */
    public static class GridConfig {
        public int dimensions;
        public long[] globalWorkSize;
        public int[] localWorkSize;
        public String pattern;

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder("GridConfig{dimensions=").append(dimensions);
            sb.append(", global=[");
            for (int i = 0; i < globalWorkSize.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(globalWorkSize[i]);
            }
            sb.append("], local=[");
            for (int i = 0; i < localWorkSize.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(localWorkSize[i]);
            }
            sb.append("], pattern=").append(pattern).append("}");
            return sb.toString();
        }
    }

    /**
     * Extract and parse grid_config from MCP JSON response.
     *
     * @param jsonResponse The full JSON response from MCP server
     * @param parameterValues Map of parameter names to their actual values
     * @return Parsed GridConfig or null if not found
     */
    public static GridConfig parseFromResponse(String jsonResponse, Map<String, Long> parameterValues) {
        // Find grid_config in JSON response
        String gridConfigJson = extractGridConfigJson(jsonResponse);
        if (gridConfigJson == null) {
            return null;
        }

        return parseGridConfig(gridConfigJson, parameterValues);
    }

    /**
     * Extract the grid_config JSON object from the response.
     */
    private static String extractGridConfigJson(String response) {
        int start = response.indexOf("\"grid_config\":");
        if (start == -1) {
            return null;
        }

        // Find the opening brace
        int jsonStart = response.indexOf("{", start);
        if (jsonStart == -1) {
            return null;
        }

        // Find matching closing brace (handle nested objects/arrays)
        int braceCount = 0;
        int jsonEnd = jsonStart;
        for (int i = jsonStart; i < response.length(); i++) {
            char c = response.charAt(i);
            if (c == '{' || c == '[') {
                braceCount++;
            } else if (c == '}' || c == ']') {
                braceCount--;
                if (braceCount == 0) {
                    jsonEnd = i + 1;
                    break;
                }
            }
        }

        return response.substring(jsonStart, jsonEnd);
    }

    /**
     * Parse the grid_config JSON and evaluate expressions.
     */
    private static GridConfig parseGridConfig(String json, Map<String, Long> parameterValues) {
        GridConfig config = new GridConfig();

        // Extract dimensions
        config.dimensions = extractInt(json, "dimensions");

        // Extract and evaluate global_work_size expressions
        String[] globalExprs = extractStringArray(json, "global_work_size");
        if (globalExprs == null) {
            globalExprs = extractStringArray(json, "global");
        }
        if (globalExprs != null) {
            config.globalWorkSize = new long[globalExprs.length];
            for (int i = 0; i < globalExprs.length; i++) {
                config.globalWorkSize[i] = evaluateExpression(globalExprs[i], parameterValues);
            }
        }

        // Extract local_work_size (these should be concrete numbers)
        int[] localSizes = extractIntArray(json, "local_work_size");
        if (localSizes == null) {
            localSizes = extractIntArray(json, "local");
        }
        config.localWorkSize = localSizes;

        // Extract pattern (optional)
        config.pattern = extractString(json, "pattern");

        return config;
    }

    /**
     * Evaluate a simple arithmetic expression with parameter substitution.
     * Supports: parameter names, integers, *, +, -, /, parentheses
     *
     * Examples:
     * - "size" -> parameter value
     * - "size*size" -> parameter value squared
     * - "numBodies" -> parameter value
     * - "1024" -> 1024
     */
    public static long evaluateExpression(String expr, Map<String, Long> params) {
        // Remove whitespace
        expr = expr.replaceAll("\\s+", "");

        // If it's just a number, return it
        try {
            return Long.parseLong(expr);
        } catch (NumberFormatException e) {
            // Not a simple number, continue parsing
        }

        // If it's a simple parameter name, look it up
        if (params.containsKey(expr)) {
            return params.get(expr);
        }

        // Handle expressions with operators
        return evaluateWithOperators(expr, params);
    }

    /**
     * Evaluate expression with operators (simple recursive descent).
     */
    private static long evaluateWithOperators(String expr, Map<String, Long> params) {
        // Handle addition/subtraction (lowest precedence)
        int parenDepth = 0;
        for (int i = expr.length() - 1; i >= 0; i--) {
            char c = expr.charAt(i);
            if (c == ')') parenDepth++;
            else if (c == '(') parenDepth--;
            else if (parenDepth == 0 && (c == '+' || c == '-') && i > 0) {
                long left = evaluateWithOperators(expr.substring(0, i), params);
                long right = evaluateWithOperators(expr.substring(i + 1), params);
                return c == '+' ? left + right : left - right;
            }
        }

        // Handle multiplication/division
        parenDepth = 0;
        for (int i = expr.length() - 1; i >= 0; i--) {
            char c = expr.charAt(i);
            if (c == ')') parenDepth++;
            else if (c == '(') parenDepth--;
            else if (parenDepth == 0 && (c == '*' || c == '/')) {
                long left = evaluateWithOperators(expr.substring(0, i), params);
                long right = evaluateWithOperators(expr.substring(i + 1), params);
                return c == '*' ? left * right : left / right;
            }
        }

        // Handle parentheses
        if (expr.startsWith("(") && expr.endsWith(")")) {
            return evaluateWithOperators(expr.substring(1, expr.length() - 1), params);
        }

        // Try as number
        try {
            return Long.parseLong(expr);
        } catch (NumberFormatException e) {
            // Not a number
        }

        // Try as parameter
        if (params.containsKey(expr)) {
            return params.get(expr);
        }

        throw new IllegalArgumentException("Cannot evaluate expression: " + expr);
    }

    // JSON parsing helpers (simple, no external dependencies)

    private static int extractInt(String json, String key) {
        Pattern pattern = Pattern.compile("\"" + key + "\"\\s*:\\s*(\\d+)");
        Matcher matcher = pattern.matcher(json);
        if (matcher.find()) {
            return Integer.parseInt(matcher.group(1));
        }
        return 0;
    }

    private static String extractString(String json, String key) {
        Pattern pattern = Pattern.compile("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
        Matcher matcher = pattern.matcher(json);
        if (matcher.find()) {
            return matcher.group(1);
        }
        return null;
    }

    private static String[] extractStringArray(String json, String key) {
        Pattern pattern = Pattern.compile("\"" + key + "\"\\s*:\\s*\\[([^\\]]+)\\]");
        Matcher matcher = pattern.matcher(json);
        if (matcher.find()) {
            String arrayContent = matcher.group(1);
            // Parse array elements (can be strings or numbers)
            String[] elements = arrayContent.split(",");
            String[] result = new String[elements.length];
            for (int i = 0; i < elements.length; i++) {
                String elem = elements[i].trim();
                // Remove quotes if present
                if (elem.startsWith("\"") && elem.endsWith("\"")) {
                    elem = elem.substring(1, elem.length() - 1);
                }
                result[i] = elem;
            }
            return result;
        }
        return null;
    }

    private static int[] extractIntArray(String json, String key) {
        Pattern pattern = Pattern.compile("\"" + key + "\"\\s*:\\s*\\[([^\\]]+)\\]");
        Matcher matcher = pattern.matcher(json);
        if (matcher.find()) {
            String arrayContent = matcher.group(1);
            String[] elements = arrayContent.split(",");
            int[] result = new int[elements.length];
            for (int i = 0; i < elements.length; i++) {
                result[i] = Integer.parseInt(elements[i].trim());
            }
            return result;
        }
        return null;
    }

    // Test
    public static void main(String[] args) {
        // Test expression evaluation
        Map<String, Long> params = new HashMap<>();
        params.put("size", 4096L);
        params.put("numBodies", 16384L);

        System.out.println("Testing expression evaluation:");
        System.out.println("size = " + evaluateExpression("size", params));
        System.out.println("size*size = " + evaluateExpression("size*size", params));
        System.out.println("numBodies = " + evaluateExpression("numBodies", params));
        System.out.println("1024 = " + evaluateExpression("1024", params));
        System.out.println("size+100 = " + evaluateExpression("size+100", params));
        System.out.println("(size*size)/4 = " + evaluateExpression("(size*size)/4", params));

        // Test JSON parsing
        String testJson = "{\"grid_config\": {\"dimensions\": 1, \"global_work_size\": [\"size*size\"], \"local_work_size\": [256], \"pattern\": \"element-wise\"}}";
        GridConfig config = parseFromResponse(testJson, params);
        if (config != null) {
            System.out.println("\nParsed grid config: " + config);
        }
    }
}
