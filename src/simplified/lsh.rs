


#[cfg(test)]
mod tests {
    use lsh_rs::prelude::LshMem;


    #[test]
    #[ignore]
    fn test_lsh() {
        let mut lsh = LshMem::new(5000, 100, 3).seed(1).srp().unwrap();
        let v1 = &[2, 3, 4];
        let v2 = &[1, 1, 1];
        lsh.store_vec(v1).unwrap();
        lsh.store_vec(v2).unwrap();
        assert!(!lsh.query_bucket(v2).unwrap().is_empty());

        let v3= &[0, 1, 4];

        let results = lsh.query_bucket(v3).unwrap();
        let bucket_len_before = results.len();
        lsh.delete_vec(v1).unwrap();
        let bucket_len_before_after = lsh.query_bucket(v1).unwrap().len();
        assert!(bucket_len_before > bucket_len_before_after);
    }
}